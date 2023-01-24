import os
import json
import itertools
from typing import Dict, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor as T
from torch.cuda.amp import GradScaler

from sentence_transformers import util
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins import DeepSpeedPlugin

from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers.optimization import AdamW, get_scheduler

from grad_cache import GradCache
from grad_cache.loss import SimpleContrastiveLoss

from src.common.utils import encode
from src.common.metrics import Metrics
from src.data.long_tokenizer import LongTokenizer
from src.models.bert import BERT, LongBERT, HierBERT
from src.models.n2vbert import Node2vecBERT
from src.models.losses import CrossEntropyLoss, TripletLoss, ScoringFunction as sf

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BiencoderModule(LightningModule):
    def __init__(self,
                 q_model_name_or_path: str, #model name (Hugging Face) or path for the query encoder.
                 d_model_name_or_path: Optional[str] = None, #model name (Hugging Face) or path for the document encoder.
                 max_document_length: Optional[int] = None, #input will be truncated to this size.
                 max_chunk_length: Optional[int] = None, #input will be split into chunks of this size.
                 pooling_mode: str = 'max', #the [CLS] representations of the chunks will be pooled with this strategy (either 'mean' or 'max').
                 loss_config: Dict[str, Union[str, float]] = {"type":"cross_entropy", "temp":0.01, "metric":"cos"}, #loss function configuration (negative log-likelihood or triplet loss).
                 lr: float = 2e-5, #learning rate.
                 scheduler_type: str = "constant_with_warmup", #type of lr scheduling among {"constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"}.
                 warmup_ratio: float = 0.01, #percentage of training steps to consider for warmup.
                 weight_decay: float = 0.0, #regularization
                 metrics_at_k: Dict[str, List[int]] = {'recall': [100, 200, 500], 'map': [100], 'mrr': [100]},
                 node_vectors_path: Optional[str] = None, #path of the pre-trained node2vec embeddings (for Node2vecBERT document encoder only).
        ):
        super().__init__()
        self.save_hyperparameters() #store all the provided arguments under the 'self.hparams' attribute.

        # Text encoders.
        self.q_encoder = BERT(model_name_or_path=q_model_name_or_path)
        self.d_encoder = HierBERT(
            model_name_or_path=d_model_name_or_path if d_model_name_or_path is not None else q_model_name_or_path,
            pooling_mode=pooling_mode,
            n_pos_chunks=int(max_document_length/max_chunk_length)+3,
        )
        # Text tokenizers.
        self.q_tokenizer = LongTokenizer(
            tokenizer_name_or_path=q_model_name_or_path,
            max_document_length=self.q_encoder.word_encoder.config.max_position_embeddings - 2 #(514 max with [CLS] and [SEP] tokens, so 512 inputs allowed).
        )
        self.d_tokenizer = LongTokenizer(
            tokenizer_name_or_path=d_model_name_or_path if d_model_name_or_path is not None else q_model_name_or_path,
            max_document_length=max_document_length,
            max_chunk_length=max_chunk_length
        )
        # Metrics scorer.
        self.scorer = Metrics(
            recall_at_k=metrics_at_k['recall'],
            map_at_k=metrics_at_k['map'],
            mrr_at_k=metrics_at_k['mrr']
        )
        # Loss module.
        self.loss = self.configure_loss(loss_config)
        

    def training_step(self, batch, batch_idx):
        q_pooled_out = self.q_encoder(batch['q_token_ids'], batch['q_attention_masks'])
        pos_d_pooled_out = self.d_encoder(batch['pos_d_token_ids'], batch['pos_d_attention_masks'], batch['pos_ids'])
        neg_d_pooled_out = self.d_encoder(batch['neg_d_token_ids'], batch['neg_d_attention_masks'], batch['neg_ids']) if batch['neg_d_token_ids'] is not None else None
        loss = self.loss(q_pooled_out, pos_d_pooled_out, neg_d_pooled_out)
        self.log('train_loss', loss, on_step=True, on_epoch=False, batch_size=self.trainer.datamodule.hparams.train_batch_size)
        return loss

    def on_validation_start(self):
        self.setup("validate")
        self._shared_start()

    def on_test_start(self):
        self.setup("test")
        self._shared_start()

    def _shared_start(self):
        # Encode all documents using current document encoder checkpoint.
        self.d_embeddings = encode(
            model=self.d_encoder,
            tokenizer=self.d_tokenizer,
            texts=self.documents, 
            text_ids=self.doc_ids, 
            batch_size=self.trainer.datamodule.hparams.eval_batch_size, 
            device=self.device, 
            show_progress=True,
        )

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val", self.val_relevant_pairs)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test", self.test_relevant_pairs)

    def _shared_step(self, batch, batch_idx, stage, relevant_pairs):
        # Encode all queries from batch using current query encoder checkpoint.
        q_embeddings = self.q_encoder(batch['q_token_ids'], batch['q_attention_masks'])

        # Retrieve top candidates -> returns a List[List[Dict[str,int]]].
        all_results = util.semantic_search(
            query_embeddings=q_embeddings, 
            corpus_embeddings=self.d_embeddings,
            top_k=max(list(itertools.chain(*self.hparams.metrics_at_k.values()))),
            score_function=self.loss.scoring_function.value['similarity'],
        )
        all_results = [[result['corpus_id']+1 for result in results] for results in all_results] #Extract the doc_id only -> List[List[int]] (NB: +1 because article ids start at 1 while semantic_search returns indices in the given list).

        # Get ground truths.
        all_ground_truths = [relevant_pairs[qid] for qid in batch['q_ids'].tolist()]

        #-------------------------------------#
        # For Error Analysis.
        #-------------------------------------#
        # import pandas as pd
        # differences = [[x for x in ground_truths if x not in predictions] for predictions, ground_truths in zip(all_results, all_ground_truths)]
        # df = pd.DataFrame(list(zip(batch['q_ids'].tolist(), all_ground_truths, differences, all_results)), columns =['q_id', 'labels', 'missed', 'predictions'])
        # df.to_csv(f'output/eda/dsr_raw_results_{batch_idx}.csv', index=False)
        #-------------------------------------#

        # Compute metrics and log them.
        scores = self.scorer.compute_all_metrics(all_ground_truths, all_results)
        for metric, value in scores.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    self.log(f"{stage}_{metric}@{k}", v, on_step=False, on_epoch=True, batch_size=self.trainer.datamodule.hparams.eval_batch_size)
            else:
                self.log(f"{stage}_{metric}", value, on_step=False, on_epoch=True, batch_size=self.trainer.datamodule.hparams.eval_batch_size)
        return scores


    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            # Calculate total number of training steps (needed for lr scheduler).
            t_num_samples = len(self.trainer.datamodule.train_dataloader().dataset)
            t_batch_size = self.trainer.datamodule.hparams.train_batch_size * max(1, self.trainer.gpus)
            a_batch_size = t_batch_size * self.trainer.accumulate_grad_batches
            self.total_steps = (t_num_samples * self.trainer.max_epochs) // a_batch_size

            # Calculate number of warmup steps (needed for lr scheduler).
            self.warmup_steps = int(self.hparams.warmup_ratio * self.total_steps)

        if stage == "validate":
            # Get dictionary of relevant (one-to-many) pairs for val set: qid -> List[doc_id].
            self.val_relevant_pairs = self.trainer.datamodule.bsard_val.one_to_many_pairs

        if stage == "test":
            # Get dictionary of relevant (one-to-many) pairs for test set: qid -> List[doc_id].
            self.test_relevant_pairs = self.trainer.datamodule.bsard_test.one_to_many_pairs

        # Get corpus of documents (same for train/val/test).
        self.documents = list(self.trainer.datamodule.dID_to_doc.values())
        self.doc_ids = list(self.trainer.datamodule.dID_to_doc.keys())


    def configure_optimizers(self):
        # Create groups of parameters to optimize (with weight decay on all parameters other than bias and layer normalization terms).
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # Configure optimizer.
        if isinstance(self.trainer.training_type_plugin, DeepSpeedPlugin):
            optimizer = DeepSpeedCPUAdam(optimizer_grouped_params, adamw_mode=True, lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-7) #1e-7 because it's the smallest value which won’t become zero when using fp16 (1e-8 will).
        else:
            optimizer = AdamW(optimizer_grouped_params, lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-7)
        # Configure learning rate scheduler.
        scheduler = get_scheduler(name=self.hparams.scheduler_type, optimizer=optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


    def configure_loss(self, config):
        assert config['type'] in ['cross_entropy', 'triplet'], f""
        assert config['metric'] in ['cos', 'dot', 'l2', 'l1'], f""
        if config['metric'] == 'cos':
            f = sf.COSINE
        elif config['metric'] == 'dot':
            f = sf.DOT
        elif config['metric'] == 'l2':
            f = sf.EUCLIDEAN
        elif config['metric'] == 'l1':
            f = sf.MANHATTAN
        if config['type'] == "cross_entropy":
            return CrossEntropyLoss(scoring_function=f, temperature=config['temp'])
        elif config['type'] == "triplet":
            return TripletLoss(scoring_function=f, margin=config['margin'])






    #-------------------------------------------------------------------#
    #                   GRADIENT CACHING (not working properly yet)
    #-------------------------------------------------------------------#
    # def training_step(self, batch, batch_idx):
    #     # Get optimizer and scheduler.
    #     optimizer = self.optimizers()
    #     scheduler = self.lr_schedulers()

    #     # Convert batch inputs to proper format.
    #     queries = dict(token_ids=batch['q_token_ids'], attention_masks=batch['q_attention_masks'])
    #     documents = dict(
    #         token_ids=torch.cat((batch['pos_d_token_ids'], batch['neg_d_token_ids']), 0) if batch['neg_d_token_ids'] is not None else batch['pos_d_token_ids'],
    #         attention_masks=torch.cat((batch['pos_d_attention_masks'], batch['neg_d_attention_masks']), 0) if batch['neg_d_attention_masks'] is not None else batch['pos_d_attention_masks']
    #     )

    #     # Compute loss and calculate the gradients with Gradient Caching (https://github.com/luyug/GradCache).
    #     loss = self.gc(queries, documents, reduction='mean')
    #     self.log('train_loss', loss, on_step=True, on_epoch=False, batch_size=self.trainer.datamodule.hparams.train_batch_size)

    #     # Perform gradient update.
    #     if self.trainer.precision == 16:
    #         # Mixed precision: not working.
    #         scaler = self.trainer.scaler
    #         scaler.unscale_(optimizer) #Unscale the gradients of optimizer's assigned params in-place.
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  #Since the gradients of optimizer's assigned params are unscaled, clips as usual.
    #         scaler.step(optimizer) #Unscale the gradients of the optimizer's assigned params. If these gradients do not contain infs or NaNs, optimizer.step() is then called. Otherwise, optimizer.step() is skipped.
    #         scaler.update() #Update the scale for next iteration.
    #         scheduler.step() #Update the learning rate.
    #     else:
    #         # FP32: gives poor results + is super slow.
    #         nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) #Clip the gradients to 1.0 to prevent the "exploding gradients" problem
    #         optimizer.step() #Update the parameters and take a step using the computed gradients.
    #         scheduler.step() #Update the learning rate.
    #     optimizer.zero_grad()
    #     return loss

    # def on_train_start(self):
    #     self.automatic_optimization = False
    #     self.gc = GradCache(
    #         models=[self.q_encoder, self.d_encoder], 
    #         chunk_sizes=16, 
    #         loss_fn=SimpleContrastiveLoss(n_hard_negatives=1),
    #         fp16=True,
    #         scaler=self.trainer.scaler,
    #     )
    #---------------------------------------#