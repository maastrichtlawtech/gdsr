import faulthandler; faulthandler.enable()
import os, sys, pathlib
import json, yaml
import argparse
import datetime as dt
from os.path import join, abspath

import wandb
import matplotlib.pyplot as plt

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

try:
    from src.pl_modules.biencoder_module import BiencoderModule
except ModuleNotFoundError:
    sys.path.append(str(pathlib.Path().resolve()))
    from src.pl_modules.biencoder_module import BiencoderModule
from src.pl_modules.gnn_module import GNNModule
from src.data.bsard_module import BSARDModule
from src.common.other import save_with_huggingface, str2bool


def main(args):
    # ------------------------ #
    #         DETERMINISM      #
    # ------------------------ #
    seed_everything(args.seed, workers=True)
    deterministic = False if args.module_to_train == 'gnn' else True  #Softmax in torch_geometric does not have a deterministic implementation yet (raises en error).
    
    # ------------------------ #
    #          WANDB           #
    # ------------------------ #
    os.makedirs(args.logs_path, exist_ok=True)
    os.environ["WANDB_DIR"] = abspath(args.logs_path)
    wandb.init(config=args, dir=args.logs_path, project="star")

    # ------------------------ #
    #         LOGGERS          #
    # ------------------------ #
    timestamp = dt.datetime.today().strftime('%Y-%m-%d_%H-%M')
    outdir = join(args.checkpoints_path, timestamp)
    wandb_logger = WandbLogger()
    tb_logger = TensorBoardLogger(
        save_dir=join(args.logs_path, 'tensorboard'), 
        name=timestamp,
        version=0,
        default_hp_metric=False,
    )

    # ------------------------ #
    #        CALLBACKS         #
    # ------------------------ #
    top_checkpoint = ModelCheckpoint(
        monitor='val_recall@200', mode="max",
        dirpath=outdir,
        filename='best_epoch-{epoch}_step-{step}_val_recall200-{val_recall@200:.2f}',
        auto_insert_metric_name=False,
        save_top_k=1,
    )
    reg_checkpoint = ModelCheckpoint(
        dirpath=outdir,
        filename='reg_epoch-{epoch}_step-{step}',
        auto_insert_metric_name=False,
        every_n_epochs=5,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if args.save_ckpt:
        checkpointing = True
        callbacks = [lr_monitor, top_checkpoint, reg_checkpoint]
    else:
        checkpointing = False
        callbacks = [lr_monitor]        

    # ------------------------ #
    #         DEEPSPEED        #
    # ------------------------ #
    # ZeRO stage 2: enable offloading of optimizer states and gradients to CPU. [Works]
    # ZeRO stage 3: enable offloading of model parameters to CPU in addition to optimizer offloading. [Fails at inference]
    if args.deepspeed:
        with open(args.deepspeed_config_path) as f:
            ds_config = json.load(f)
        ds_config["fp16"]["enabled"] = args.fp16
        deepspeed_strategy = DeepSpeedPlugin(
            config=ds_config, 
            logging_batch_size_per_gpu=args.train_batch_size
        )
    else:
        deepspeed_strategy = None

    # ------------------------ #
    #           MODEL          #
    # ------------------------ #
    if args.module_to_train == 'biencoder':
        model = BiencoderModule(
            q_model_name_or_path=args.q_model_name_or_path,
            d_model_name_or_path=args.d_model_name_or_path,
            node_vectors_path=args.node_vectors_path,
            max_document_length=args.max_document_length,
            max_chunk_length=args.max_chunk_length,
            pooling_mode=args.pooling_mode,
            loss_config=args.loss_config,
            lr=args.lr,
            scheduler_type=args.scheduler,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
        )
    elif args.module_to_train == 'gnn':
        biencoder = BiencoderModule.load_from_checkpoint(args.biencoder_ckpt)
        model = GNNModule(
            c_in=biencoder.d_encoder.dim,
            c_hidden=biencoder.d_encoder.dim * 4,
            c_out=biencoder.d_encoder.dim,
            layer_name=args.gnn_layer_name,
            num_layers=args.num_gnn_layers,
            loss_config=args.loss_config,
            lr=args.lr,
            scheduler_type=args.scheduler,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
        )

    # ------------------------ #
    #        DATAMODULE        #
    # ------------------------ #
    dm = BSARDModule(
        target_module=args.module_to_train,
        biencoder=biencoder if args.module_to_train == 'gnn' else model,
        train_queries_filepath=args.train_queries_path,
        val_queries_filepath=args.val_queries_path,
        test_queries_filepath=args.test_queries_path,
        synthetic_queries_filepath=args.synthetic_queries_path,
        documents_filepath=args.documents_path,
        nodes_filepath=args.nodes_path,
        edges_filepath=args.edges_path,
        hard_negatives_filepath=args.hard_negatives_path,
        add_doc_title=args.add_doc_title,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ------------------------ #
    #         TRAINER          #
    # ------------------------ #
    trainer = Trainer(
        accelerator="auto",                                    #
        gpus=torch.cuda.device_count(),                        #
        strategy=deepspeed_strategy,                           #
        precision=16 if args.fp16 else 32,                     #
        limit_train_batches=args.ratio_train_set_to_use,       #
        max_epochs=args.epochs,                                #
        gradient_clip_val=args.gradient_clip_val,              #
        accumulate_grad_batches=args.accumulate_grad_batches,  #
        limit_val_batches=1.0 if args.do_val else 0.0,         #
        check_val_every_n_epoch=1,                             #
        num_sanity_val_steps=0,                                #
        logger=[tb_logger, wandb_logger],                      #
        log_every_n_steps=1,                                   #
        callbacks=callbacks,                                   #
        enable_checkpointing=checkpointing,                    #
        deterministic=deterministic,                           #
        #detect_anomaly=True,                                  #
    )

    # ------------------------ #
    #   LEARNING RATE FINDER   #
    # ------------------------ #
    if args.find_lr:
        lr_finder = trainer.tuner.lr_find(
            model=model, 
            datamodule=dm, 
            min_lr=1e-8, 
            max_lr=1e-3,
        )
        model.hparams.lr = lr_finder.suggestion()
        fig = lr_finder.plot(suggest=True)
        fig.savefig(join(args.checkpoints_path, 'lr_finder.pdf'))

    # ------------------------ #
    #         TRAINING         #
    # ------------------------ #
    if args.do_train:
        trainer.fit(model, datamodule=dm)

    # ------------------------- #
    #   CHECKPOINTING IN FP32   #
    # ------------------------- #
    ckpt_paths = {
        'last': join(outdir, 'last.ckpt'),
        'best': trainer.checkpoint_callback.best_model_path,
    }
    if args.save_ckpt:
        if isinstance(trainer.training_type_plugin, DeepSpeedPlugin):
            for ckpt_name, fp16_path in ckpt_paths.items():
                fp32_path = join(fp16_path, f"{ckpt_name}_fp32.ckpt")
                convert_zero_checkpoint_to_fp32_state_dict(fp16_path, fp32_path)
                ckpt_paths[ckpt_name] = fp32_path
        if args.module_to_train == 'biencoder':
            for _, fp32_path in ckpt_paths.items():
                save_with_huggingface(fp32_path)

    # ------------------------ #
    #         TESTING          #
    # ------------------------ #
    if args.do_test:
        # torch.distributed.destroy_process_group()
        # if trainer.global_rank == 0:
        trainer = Trainer(accelerator="auto", gpus=torch.cuda.device_count(), logger=False)
        for ckpt_name, fp32_path in ckpt_paths.items():
            results = trainer.test(model, datamodule=dm, ckpt_path=fp32_path)
            with open(join(abspath(fp32_path + '/../'), f'{ckpt_name}_test_results.json'), 'w') as fOut:
                json.dump(results, fOut, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data paths.
    parser.add_argument("--train_queries_path", type=str, help="The training data file path.")
    parser.add_argument("--val_queries_path", type=str, help="The validation data file path.")
    parser.add_argument("--test_queries_path", type=str, help="The testing data file path.")
    parser.add_argument("--synthetic_queries_path", type=str, help="The synthetic data file path.")
    parser.add_argument("--documents_path", type=str, help="The documents collection file path.")
    parser.add_argument("--hard_negatives_path", type=str, help="The hard negatives file path.")
    parser.add_argument("--logs_path", type=str, help="Folder to save logs during training.")
    parser.add_argument("--checkpoints_path", type=str, help="Folder to save checkpoints during training.")
    # Biencoder parameters.
    parser.add_argument("--q_model_name_or_path", type=str, help="The query encoder checkpoint for weights initialization.")
    parser.add_argument("--d_model_name_or_path", type=str, help="The document encoder checkpoint for weights initialization.")
    parser.add_argument("--max_chunk_length", type=int, help="Length of chunks.")
    parser.add_argument("--max_document_length", type=int, help="Maximum length at which the documents will be truncated.")
    parser.add_argument("--pooling_mode", type=str, choices=('mean', 'max', 'None'), help="Type of pooling to perform on the [CLS] chunk representations from long documents.")
    parser.add_argument("--add_doc_title", type=str2bool, help="Whether or not to append the article headings in front of their content.")
    # GNN parameters.
    parser.add_argument("--biencoder_ckpt", type=str, help="A trained biencoder checkpoint used to initialized thenode features.")
    parser.add_argument("--nodes_path", type=str, help="The BSARD nodes file path.")
    parser.add_argument("--edges_path", type=str, help="The BSARD edges file path.")
    parser.add_argument("--gnn_layer_name", type=str, help="The type of GNN layer to use.")
    parser.add_argument("--num_gnn_layers", type=int, help="The number of GNN layers to use.")
    parser.add_argument("--node_vectors_path", type=str, help="The BSARD node2vec representations file path.")
    # Training hyper-parameters.
    parser.add_argument("--module_to_train", type=str, choices=('biencoder', 'gnn'), help="Name of the Pytorch Lightning module to train.")
    parser.add_argument("--loss_config", type=yaml.safe_load, help="Training loss configuration dictionary. Should either be {'type':'cross_entropy','temp':0.01,'metric':'cos'} or {'type':'triplet','margin':1.0,'metric':'l2'}.")
    parser.add_argument("--train_batch_size", type=int, help="The batch size per GPU/TPU core/CPU for training.")
    parser.add_argument("--eval_batch_size", type=int, help="The batch size per GPU/TPU core/CPU for evaluation.")
    parser.add_argument("--epochs", type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--lr", type=float, help="The initial learning rate for AdamW optimizer.")
    parser.add_argument("--scheduler", type=str, choices=("constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"), help="Learning rate scheduler.")
    parser.add_argument("--warmup_ratio", type=float, help="Ratio of total training steps used for a linear warmup from 0 to 'lr'.")
    parser.add_argument("--weight_decay", type=float, help="The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.")
    parser.add_argument("--gradient_clip_val", type=float, help="Maximum gradient norm (for gradient clipping).")
    parser.add_argument("--accumulate_grad_batches", type=int, help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--fp16", type=str2bool, help="Whether or not to use mixed precision during training.")
    parser.add_argument("--deepspeed", type=str2bool, help="Whether or not to use deepspeed optimization during training.")
    parser.add_argument("--deepspeed_config_path", type=str, help="Path to the deepspeed configuration file.")
    parser.add_argument("--seed", type=int, help="Random seed that will be set at the beginning of training.")
    # Other.
    parser.add_argument("--do_train", type=str2bool, help="Wether or not to perform training.")
    parser.add_argument("--ratio_train_set_to_use", type=float, help="How much of training dataset to check.")
    parser.add_argument("--do_val", type=str2bool, help="Wether or not to perform validation during training.")
    parser.add_argument("--do_test", type=str2bool, help="Wether or not to perform testing after training.")
    parser.add_argument("--num_workers", type=int, help="Number of subprocesses to use for data loading.")
    parser.add_argument("--find_lr", type=str2bool, help="Wether or not to automatically find the optimal learning rate before training.")
    parser.add_argument("--save_ckpt", type=str2bool, help="Wether or not to save model checkpoints during/after training.")
    # Parse.
    args, _ = parser.parse_known_args()
    main(args)
