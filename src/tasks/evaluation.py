import os, sys, pathlib
import json
import argparse
from os.path import join, abspath

import torch
from pytorch_lightning import Trainer
from transformers import AutoTokenizer, AutoModel

try:
    from src.data.bsard_module import BSARDModule
except ModuleNotFoundError:
    # Add project root directory to the PYTHONPATH environment variable.
    sys.path.append(str(pathlib.Path().resolve()))
    from src.data.bsard_module import BSARDModule
from src.data.long_tokenizer import LongTokenizer
from src.pl_modules.gnn_module import GNNModule
from src.pl_modules.biencoder_module import BiencoderModule


def main(args):
    if args.module_to_test == 'gnn':
        biencoder = BiencoderModule.load_from_checkpoint(args.biencoder_ckpt_path)
        model = GNNModule.load_from_checkpoint(args.gnn_ckpt_path)
        ckpt_path = args.gnn_ckpt_path
    else:
        model = BiencoderModule.load_from_checkpoint(args.biencoder_ckpt_path)
        ckpt_path = args.biencoder_ckpt_path
    dm = BSARDModule(
        target_module=args.module_to_test,
        biencoder=biencoder if args.module_to_test == 'gnn' else model,
        test_queries_filepath=args.queries_path,
        documents_filepath=args.documents_path,
        nodes_filepath=args.nodes_path,
        edges_filepath=args.edges_path,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    trainer = Trainer(accelerator="auto", gpus=torch.cuda.device_count(), logger=False)
    results = trainer.test(model, datamodule=dm)
    # with open(join(abspath(ckpt_path + '/../'), f'{args.out_filename}.json'), 'w') as fOut:
    #     json.dump(results, fOut, indent=2)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data paths.
    parser.add_argument("--queries_path", type=str, help="The evaluation data file path.")
    parser.add_argument("--documents_path", type=str, help="The documents collection file path.")
    parser.add_argument("--nodes_path", type=str, help="The BSARD nodes file path.")
    parser.add_argument("--edges_path", type=str, help="The BSARD edges file path.")
    # Model to test and checkpoints.
    parser.add_argument("--module_to_test", type=str, choices=('biencoder', 'gnn'), help="Name of the Pytorch Lightning module to train.")
    parser.add_argument("--biencoder_ckpt_path", type=str, help="Path of the trained biencoder model.")
    parser.add_argument("--gnn_ckpt_path", type=str, help="Path of the trained GNN model.")
    # Other.
    parser.add_argument("--eval_batch_size", type=int, help="The batch size per GPU/TPU core/CPU for evaluation.")
    parser.add_argument("--seed", type=int, help="Random seed that will be set at the beginning of training.")
    parser.add_argument("--num_workers", type=int, help="Number of subprocesses to use for data loading.")
    parser.add_argument("--out_filename", type=str, help="Name of the output file.")
    args, _ = parser.parse_known_args()
    main(args)
