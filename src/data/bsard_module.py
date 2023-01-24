import json
from typing import Type, Optional, List, Tuple

import torch
from torch import nn, Tensor as T
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import pytorch_lightning as pl
from torch_geometric.utils.subgraph import k_hop_subgraph

from src.common import utils
from src.data.bsard_dataset import BSARDataset
from src.data.bsard_graph import BSARDGraph


class BSARDModule(pl.LightningDataModule):
    def __init__(self,
                 biencoder: Type[pl.LightningModule], #biencoder model (needed to tokenize or encode queries and documents).
                 target_module: str, #whether we optimize the 'biencoder' or the 'GNN' on top of the (frozen) document encoder.
                 documents_filepath: str, #path of the document corpus.
                 train_queries_filepath: Optional[str] = None, #path of the training samples.
                 val_queries_filepath: Optional[str] = None, #path of the validation samples.
                 test_queries_filepath: Optional[str] = None, #path of the testing samples.
                 synthetic_queries_filepath: Optional[str] = None, #path of the synthetic samples.
                 nodes_filepath: Optional[str] = None, #path of the nodes (documents + headings). Only needed if we optimize the GNN module.
                 edges_filepath: Optional[str] = None, #path of the edges between nodes. Only needed if we optimize the GNN module.
                 hard_negatives_filepath: Optional[str] = None, #path of the hard negative documents ids for the training samples.
                 add_doc_title: bool = False, #whether or not we should append the document title before its content.
                 train_batch_size: int = 22, #
                 eval_batch_size: int = 512, #
                 seed: int = 42, #global seed.
                 num_workers: int = 0, #number of parallel workers for the dataloader.
        ):
        super().__init__()
        assert target_module in ['biencoder', 'gnn'], f"ERROR: Unknown 'target_module': {target_module}."
        self.save_hyperparameters(ignore='biencoder')
        self.biencoder = biencoder


    def setup(self, stage: Optional[str] = None):
        # Load corpus of articles.
        documents = pd.read_csv(self.hparams.documents_filepath)

        if stage == "fit":
            # Load train/val sets. If no validation data is given, split training set into 80/20 train/val sets.
            train_queries = pd.read_csv(self.hparams.train_queries_filepath)
            if str(self.hparams.val_queries_filepath) == 'None':
                train_queries, val_queries = self.split_train_val(train_queries, train_frac=0.8)
            else:
                val_queries = pd.read_csv(self.hparams.val_queries_filepath)

            # If synthetic queries are given, add them to the training set.
            if str(self.hparams.synthetic_queries_filepath) != 'None':
                synthetic_queries = pd.read_csv(self.hparams.synthetic_queries_filepath)
                train_queries = pd.concat([train_queries, synthetic_queries], ignore_index=True)

            # Remove questions in train that also appear in dev, if any.
            dup_questions = train_queries[train_queries['question'].isin(val_queries['question'])]
            if len(dup_questions) > 0:
                print(f"Found {len(dup_questions)} questions that appear both in train and dev sets. Removing them from train set...")
                train_queries.drop(dup_questions.index, inplace=True)
            
            # Load hard negatives for training questions if file exists.
            hard_negatives = None
            if self.hparams.hard_negatives_filepath is not None:
                with open(self.hparams.hard_negatives_filepath) as f:
                    hard_negatives = json.load(f)

            # Create the train/val Pytorch datasets for use in dataloaders.
            self.bsard_train = BSARDataset(train_queries, documents, 'train', self.hparams.add_doc_title, hard_negatives)
            self.bsard_val = BSARDataset(val_queries, documents, 'dev', self.hparams.add_doc_title)

            # Get ID to content mappings.
            self.qID_to_query = {**self.bsard_train.queries, **self.bsard_val.queries}
            self.dID_to_doc = self.bsard_train.documents

        if stage == "test":
            # Create the test Pytorch dataset for use in dataloaders.
            test_queries = pd.read_csv(self.hparams.test_queries_filepath)
            self.bsard_test = BSARDataset(test_queries, documents, 'test', self.hparams.add_doc_title)

            # Get ID to content mappings.
            self.qID_to_query = self.bsard_test.queries
            self.dID_to_doc = self.bsard_test.documents

        if self.hparams.target_module == 'gnn':
            # Pre-compute the query embeddings.
            self.q_embeddings = utils.encode(
                tokenizer=self.biencoder.q_tokenizer, 
                model=self.biencoder.q_encoder, 
                texts=list(self.qID_to_query.values()),
                batch_size=256,
            )
            self.qID_to_vecID = dict(map(lambda t: (t[1], t[0]), enumerate(self.qID_to_query.keys())))

            # Pre-compute the PyG graph of articles.
            self.bsard_graph = BSARDGraph(
                articles_filepath=self.hparams.documents_filepath,
                nodes_filepath=self.hparams.nodes_filepath,
                edges_filepath=self.hparams.edges_filepath,
            )
            self.G, self.dID_to_nodeID = self.bsard_graph.build_pyg_graph(
                tokenizer=self.biencoder.d_tokenizer,
                encoder=self.biencoder.d_encoder,
                artID_to_article=self.dID_to_doc,
            )
        

    def train_dataloader(self):
        return DataLoader(self.bsard_train, batch_size=self.hparams.train_batch_size, num_workers=self.hparams.num_workers, collate_fn=self.collate_train)


    def val_dataloader(self):
        return DataLoader(self.bsard_val, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, collate_fn=self.collate_eval)


    def test_dataloader(self):
        return DataLoader(self.bsard_test, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, collate_fn=self.collate_eval)


    def split_train_val(self, df: Type[pd.DataFrame], train_frac: float):
        # Extract the duplicated questions to put them in the training set only.
        duplicates = df[df.duplicated(['question'], keep=False)]
        uniques = df.drop(duplicates.index)

        # Compute the fraction of unique questions to place in training set so that these questions complemented by the duplicates sums up to the given 'train_frac' ratio.
        train_frac_unique = (train_frac * df.shape[0] - duplicates.shape[0]) / uniques.shape[0]

        # Split the unique questions in train/val sets accordingly.
        train_unique = uniques.sample(frac=train_frac_unique, random_state=self.hparams.seed)
        val = uniques.drop(train_unique.index).sample(frac=1.0, random_state=self.hparams.seed)

        # Add the duplicated questions to the training set, reset indexes, and return.
        train = (pd.concat([train_unique, duplicates])
                  .sample(frac=1.0, random_state=self.hparams.seed)
                  .reset_index(drop=True))
        val = val.reset_index(drop=True)
        return train, val


    def collate_train(self, batch: List[Tuple[str, str]]) -> Tuple[T, T, T, T]:
        """The collate_fn function gets called with a list of return values from your Datasest.__getitem__(), and should return stacked tensors.
        In this case, the batch is a list of tuples: [(qid, query, pos_doc, neg_doc), ...].
        """
        # Unzip a list of tuples into individual lists: [q_id, ...], [pos_id, ...], [neg_id, ...], [query, ...], [pos_doc, ...], [neg_doc, ...].
        q_ids, pos_ids, neg_ids, queries, pos_documents, neg_documents = map(list, zip(*batch))

        if self.hparams.target_module == 'biencoder':
            # Convert q_ids, pos_ids and neg_ids to proper torch type.
            q_ids = torch.tensor(q_ids, dtype=torch.int)
            pos_ids = torch.tensor(pos_ids, dtype=torch.int)
            neg_ids = torch.tensor(neg_ids, dtype=torch.int) if not all([idx is None for idx in neg_ids]) else None

            # Tokenize the queries and documents, and return.
            q_token_ids, q_attention_masks = self.biencoder.q_tokenizer(queries)
            pos_d_token_ids, pos_d_attention_masks = self.biencoder.d_tokenizer(pos_documents)
            neg_d_token_ids, neg_d_attention_masks = self.biencoder.d_tokenizer(neg_documents) if not all([doc is None for doc in neg_documents]) else (None, None)

            b = {
                'q_ids': q_ids, 'q_token_ids': q_token_ids, 'q_attention_masks': q_attention_masks, 
                'pos_ids': pos_ids, 'pos_d_token_ids': pos_d_token_ids, 'pos_d_attention_masks': pos_d_attention_masks,
                'neg_ids': neg_ids, 'neg_d_token_ids': neg_d_token_ids, 'neg_d_attention_masks': neg_d_attention_masks,
            }
        elif self.hparams.target_module == 'gnn':
            # Get indices of feature vectors corresponding to query/document IDs.
            q_idx = torch.tensor([self.qID_to_vecID.get(qid) for qid in q_ids], dtype=torch.int)
            pos_idx = torch.tensor([self.dID_to_nodeID.get(pos_id) for pos_id in pos_ids], dtype=torch.int)
            neg_idx = torch.tensor([self.dID_to_nodeID.get(neg_id) for neg_id in neg_ids], dtype=torch.int) if not all([i is None for i in neg_ids]) else None
            doc_idx = torch.cat((pos_idx, neg_idx)) if neg_idx is not None else pos_idx

            # Get embeddings from queries in the batch.
            q_features = torch.index_select(input=self.q_embeddings, index=q_idx, dim=0)
            
            # Sample subgraph with two-hop neighbors given documents in the batch.
            subgraph_node_ids, subgraph_edge_index, loc_doc_idx, _ = k_hop_subgraph(
                node_idx=doc_idx.long(),
                edge_index=self.G.edge_index,
                num_hops=2,
                relabel_nodes=True,
            )
            subgraph_node_features = torch.index_select(input=self.G.x, index=subgraph_node_ids, dim=0)

            # Get mappings from doc IDs to node IDs from sampled subgraph.
            docID_to_nodeIDsubgraph = torch.cat((doc_idx, loc_doc_idx)).view(2, -1)
            if neg_idx is None:
                posID_to_nodeIDsubgraph, negID_to_nodeIDsubgraph = docID_to_nodeIDsubgraph, None
            else:
                posID_to_nodeIDsubgraph, negID_to_nodeIDsubgraph = torch.split(
                    docID_to_nodeIDsubgraph,
                    split_size_or_sections=[pos_idx.size(0), neg_idx.size(0)], 
                    dim=1,
                )
            b = {
                'q_ids': torch.tensor(q_ids, dtype=torch.int), 'q_features': q_features,
                'node_features': subgraph_node_features, 'edge_index': subgraph_edge_index,
                'posID_to_nodeID': posID_to_nodeIDsubgraph, 'negID_to_nodeID': negID_to_nodeIDsubgraph,
            }
        return b


    def collate_eval(self, batch: List[Tuple[str, str]]) -> Tuple[T, T, T, T]:
        # Unzip a list of tuples into individual lists: [q_id, ...], [pos_id, ...], [neg_id, ...], [query, ...], [pos_doc, ...], [neg_doc, ...].
        q_ids, _, _, queries, _, _ = map(list, zip(*batch))

        if self.hparams.target_module == 'biencoder':
            q_ids = torch.tensor(q_ids, dtype=torch.int)
            q_token_ids, q_attention_masks = self.biencoder.q_tokenizer(queries)
            b = {'q_ids': q_ids, 'q_token_ids': q_token_ids, 'q_attention_masks': q_attention_masks}
        
        elif self.hparams.target_module == 'gnn':
            q_idx = torch.tensor([self.qID_to_vecID.get(qid) for qid in q_ids], dtype=torch.int)
            q_features = torch.index_select(input=self.q_embeddings, index=q_idx, dim=0)
            b = {'q_ids': torch.tensor(q_ids, dtype=torch.int), 'q_features': q_features}
        
        return b
