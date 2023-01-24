import os
import json
import itertools
from typing import Dict, List, Optional, Type, Union
from rich import print, inspect, pretty
pretty.install()

import torch
import torch.nn.functional as F
from torch import nn, Tensor as T

from sentence_transformers import util
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins import DeepSpeedPlugin

from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers.optimization import AdamW, get_scheduler

from src.models.gnn import GNN
from src.common.utils import encode
from src.common.metrics import Metrics
from src.models.losses import CrossEntropyLoss, TripletLoss, ScoringFunction as sf

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GNNModule(LightningModule):
    def __init__(self,
                 c_in: int, #Dimension of input features.
                 c_hidden: int, #Dimension of hidden features.
                 c_out: int, #Dimension of the output features.
                 layer_name: str = "GATv2", #String of the graph layer to use.
                 num_layers: int = 2, #Number of "hidden" graph layers.
                 loss_config: Dict[str, Union[str, float]] = {"type":"cross_entropy", "temp":0.01, "metric":"cos"}, #loss function configuration (negative log-likelihood or triplet loss).
                 lr: float = 2e-5, #learning rate.
                 scheduler_type: str = "constant_with_warmup", #type of lr scheduling among {"constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"}.
                 warmup_ratio: float = 0.01, #percentage of training steps to consider for warmup.
                 weight_decay: float = 0.0, #regularization
                 metrics_at_k: Dict[str, List[int]] = {'recall': [100, 200, 500], 'map': [100], 'mrr': [100]}
        ):
        super().__init__()
        self.save_hyperparameters() #store all the provided arguments under the 'self.hparams' attribute.
        self.gnn = GNN(c_in=c_in, c_hidden=c_hidden, c_out=c_out, layer_name=layer_name, num_layers=num_layers)
        self.scorer = Metrics(recall_at_k=metrics_at_k['recall'], map_at_k=metrics_at_k['map'], mrr_at_k=metrics_at_k['mrr'])
        self.loss = self.configure_loss(loss_config)

    def forward(self, node_features, edge_index):
        return self.gnn(node_features, edge_index)
        
    def training_step(self, batch, batch_idx):
        out = self(batch['node_features'], batch['edge_index'])
        pos_features = torch.index_select(input=out, index=batch['posID_to_nodeID'][1], dim=0)
        neg_features = torch.index_select(input=out, index=batch['negID_to_nodeID'][1], dim=0) if batch['negID_to_nodeID'] is not None else None
        loss = self.loss(batch['q_features'], pos_features, neg_features)
        self.log('train_loss', loss, on_step=True, on_epoch=False, batch_size=self.trainer.datamodule.hparams.train_batch_size)
        return loss

    def on_validation_start(self):
        self._shared_start()

    def on_test_start(self):
        self._shared_start()

    def _shared_start(self):
        updated_node_features = self(
            self.trainer.datamodule.G.x.to(device=self.device, dtype=self.dtype),
            self.trainer.datamodule.G.edge_index.to(self.device),
        )
        self.d_embeddings = torch.index_select(
            input=updated_node_features, 
            index=torch.tensor(list(self.trainer.datamodule.dID_to_nodeID.values()), dtype=torch.int, device=self.device), 
            dim=0,
        )
        self.corpusID_to_dID = dict(map(lambda t: (t[0], t[1]), enumerate(self.trainer.datamodule.dID_to_nodeID.keys())))

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val", self.val_relevant_pairs)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test", self.test_relevant_pairs)

    def _shared_step(self, batch, batch_idx, stage, relevant_pairs):
        all_results = util.semantic_search(
            query_embeddings=batch['q_features'], 
            corpus_embeddings=self.d_embeddings,
            top_k=max(list(itertools.chain(*self.hparams.metrics_at_k.values()))),
            score_function=self.loss.scoring_function.value['similarity'],
        )
        all_results = [[self.corpusID_to_dID.get(result['corpus_id']) for result in results] for results in all_results] #Extract the doc_id only -> List[List[int]].
        all_ground_truths = [relevant_pairs[qid] for qid in batch['q_ids'].tolist()]

        #-------------------------------------#
        # For Error Analysis.
        #-------------------------------------#
        # import pandas as pd
        # differences = [[x for x in ground_truths if x not in predictions] for predictions, ground_truths in zip(all_results, all_ground_truths)]
        # df = pd.DataFrame(list(zip(batch['q_ids'].tolist(), all_ground_truths, differences, all_results)), columns =['q_id', 'labels', 'missed', 'predictions'])
        # df.to_csv(f'output/eda/gnn_raw_results_{batch_idx}.csv', index=False)
        #-------------------------------------#
        
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

            # Get dictionary of relevant (one-to-many) pairs for val set.
            self.val_relevant_pairs = self.trainer.datamodule.bsard_val.one_to_many_pairs #qid -> List[doc_id]

        if stage == "validate":
            # Get dictionary of relevant (one-to-many) pairs for val set.
            self.val_relevant_pairs = self.trainer.datamodule.bsard_val.one_to_many_pairs #qid -> List[doc_id]

        if stage == "test":
            # Get dictionary of relevant (one-to-many) pairs for test set.
            self.test_relevant_pairs = self.trainer.datamodule.bsard_test.one_to_many_pairs

        # Get corpus of documents (same for train/val/test).
        self.documents = list(self.trainer.datamodule.dID_to_doc.values())
        self.doc_ids = list(self.trainer.datamodule.dID_to_doc.keys())

    def configure_optimizers(self):
        # Configure optimizer.
        if isinstance(self.trainer.training_type_plugin, DeepSpeedPlugin):
            optimizer = DeepSpeedCPUAdam(
                model_params=self.parameters(), 
                adamw_mode=True, 
                lr=self.hparams.lr, 
                betas=(0.9, 0.999), 
                eps=1e-7, #1e-7 because it's the smallest value which won’t become zero when using fp16 (1e-8 will).
                weight_decay = self.hparams.weight_decay,
            )
        else:
            optimizer = AdamW(
                params=self.parameters(),
                 lr=self.hparams.lr, 
                 betas=(0.9, 0.999), 
                 eps=1e-7,
                 weight_decay = self.hparams.weight_decay,
            )
        # Configure learning rate scheduler.
        scheduler = get_scheduler(name=self.hparams.scheduler_type, optimizer=optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def configure_loss(self, config):
        if config['type'] == "cross_entropy":
            return CrossEntropyLoss(
                scoring_function=sf.COSINE if config['metric'] == 'cos' else sf.DOT, 
                temperature=config['temp']
            )
        elif config['type'] == "triplet":
            return TripletLoss(
                scoring_function=sf.EUCLIDEAN if config['metric'] == 'l2' else sf.MANHATTAN, 
                margin=config['margin']
            )
