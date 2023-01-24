import pytorch_lightning as pl
from datasets import load_dataset
from typing import Type, List, Tuple
from torch.utils.data import DataLoader


class mMARCOModule(pl.LightningDataModule):
    """RuntimeError: stack expects each tensor to be equal size, but got [33] at entry 0 and [31] at entry 1.
    stack expects each tensor to be equal size, but got [33] at entry 0 and [31] at entry 1
    """
    def __init__(self,
                 biencoder: Type[pl.LightningModule], #biencoder model (needed to tokenize or encode queries and documents).
                 language: str, #language to load mMARCO.
                 train_batch_size: int, #training batch size.
                 num_workers: int, #number of parallel workers for the dataloader.
        ):
        super().__init__()
        self.save_hyperparameters(ignore='biencoder')
        self.biencoder = biencoder

    def setup(self, stage: str):
        self.dataset = load_dataset("unicamp-dl/mmarco", self.hparams.language)
        self.dataset['train'] = self.dataset['train'].map(self.convert_to_features, batched=True, remove_columns=['query', 'positive', 'negative'])
        self.dataset['train'].set_format(type="torch", columns=[c for c in self.dataset['train'].column_names if c not in ['q_ids', 'pos_ids', 'neg_ids']])

    def convert_to_features(self, batch, indices=None):
        batch_size = len(batch['query'])
        q_token_ids, q_attention_masks = self.biencoder.q_tokenizer(batch['query'])
        pos_d_token_ids, pos_d_attention_masks = self.biencoder.d_tokenizer(batch['positive'])
        neg_d_token_ids, neg_d_attention_masks = self.biencoder.d_tokenizer(batch['negative'])
        b = {
            'q_ids': [None]*batch_size, 'q_token_ids': q_token_ids.detach().cpu().numpy(), 'q_attention_masks': q_attention_masks.detach().cpu().numpy(), 
            'pos_ids': [None]*batch_size, 'pos_d_token_ids': pos_d_token_ids.detach().cpu().numpy(), 'pos_d_attention_masks': pos_d_attention_masks.detach().cpu().numpy(),
            'neg_ids': [None]*batch_size, 'neg_d_token_ids': neg_d_token_ids.detach().cpu().numpy(), 'neg_d_attention_masks': neg_d_attention_masks.detach().cpu().numpy(),
        }
        return b

    def prepare_data(self):
        load_dataset("unicamp-dl/mmarco", self.hparams.language) #39,780,811 training samples of type ['query', 'positive', 'negative'].

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.hparams.train_batch_size, num_workers=self.hparams.num_workers, shuffle=True)
