import math
import numpy as np
from typing import List, Tuple, Union, Optional

import torch
from torch import nn, Tensor as T
from transformers import AutoModel, logging as hf_logger
hf_logger.set_verbosity_error()

from src.models.pooling import Pooling


class BERT(nn.Module):
    def __init__(self, model_name_or_path: str):
        super(BERT, self).__init__()
        self.word_encoder = AutoModel.from_pretrained(model_name_or_path)
        self.dim = self.word_encoder.config.hidden_size

    def forward(self, token_ids: T, attention_masks: T, input_ids: Optional[T] = None) -> Tuple[T, T, T]:
        """
        Args:
            token_ids: 2D tensor of size [batch_size, max_chunk_length].
            attention masks: 2D tensor of size [batch_size, max_chunk_length].
            input_ids: 1D tensor of size [batch_size]. Optional: unique IDs of input texts.
        Returns:
            text_representations: 2D tensor of size [batch_size, embedding_dim].
        """
        token_embeddings = self.word_encoder(input_ids=token_ids, attention_mask=attention_masks, return_dict=False)[0]
        token_embeddings = token_embeddings[:,0,:] # Keep only [CLS] representations.
        return token_embeddings


class LongBERT(nn.Module):
    def __init__(self, model_name_or_path: str, pooling_mode: str):
        super(LongBERT, self).__init__()
        self.word_encoder = AutoModel.from_pretrained(model_name_or_path)
        self.pooler = Pooling(pooling_mode)
        self.dim = self.word_encoder.config.hidden_size

    def forward(self, token_ids: T, attention_masks: T, input_ids: Optional[T] = None) -> Tuple[T, T, T]:
        """
        Args:
            token_ids: 3D tensor of size [batch_size, batch_max_chunks, max_chunk_length].
            attention masks: 3D tensor of size [batch_size, batch_max_chunks, max_chunk_length].
            input_ids: 1D tensor of size [batch_size]. Optional: unique IDs of input texts.
        Returns:
            text_representations: 2D tensor of size [batch_size, embedding_dim].
        """
        # Reshape to 2D tensor of size [batch_size x max_chunks, max_chunk_length].
        token_ids_reshaped = token_ids.contiguous().view(-1, token_ids.size(-1))
        attention_masks_reshaped = attention_masks.contiguous().view(-1, attention_masks.size(-1))

        # Compute context-aware word embeddings with the BERT-based encoder.
        token_embeddings = self.word_encoder(input_ids=token_ids_reshaped, attention_mask=attention_masks_reshaped, return_dict=False)[0]

        # Reshape back to 4D tensor of size [batch_size, batch_max_chunks, max_chunk_length, hidden_size]
        token_embeddings = token_embeddings.contiguous().view(*tuple(token_ids.size()), self.word_encoder.config.hidden_size)

        # Keep only [CLS] embeddings (and corresponding attention masks) for each chunk in the batch.
        token_embeddings = token_embeddings[:,:,0,:] #-> 3D tensor of size [batch_size, batch_max_chunks, hidden_size]
        attention_masks = attention_masks[:,:,0] #-> 2D tensor of size [batch_size, batch_max_chunks]

        # Pool the [CLS] chunk representations to distill a global representation for each document in the batch.
        text_representations = self.pooler(token_embeddings, attention_masks)
        return text_representations


class HierBERT(nn.Module):
    def __init__(self, model_name_or_path: str, pooling_mode: str, n_pos_chunks: int):
        super(HierBERT, self).__init__()
        # Word-wise BERT-based encoder.
        self.word_encoder = AutoModel.from_pretrained(model_name_or_path)
        self.dim = self.word_encoder.config.hidden_size

        # Chunk-wise sinusoidal positional embeddings.
        self.chunk_pos_embeddings = nn.Embedding(num_embeddings=n_pos_chunks+1, 
                                                 embedding_dim=self.dim,
                                                 padding_idx=0,
                                                 _weight=HierBERT.sinusoidal_init(n_pos_chunks+1, self.dim, 0))
        # Chunk-wise Transformer-based encoder.
        self.chunk_encoder = nn.Transformer(
            num_encoder_layers=2,
            num_decoder_layers=0,
            d_model=self.dim,               #self.word_encoder.config.hidden_size
            dim_feedforward=self.dim*4,     #self.word_encoder.config.intermediate_size,
            nhead=12,                       #self.word_encoder.config.num_attention_heads,
            activation='gelu',              #self.word_encoder.config.hidden_act,
            dropout=0.1,                    #self.word_encoder.config.hidden_dropout_prob,
            layer_norm_eps=1e-5,            #self.word_encoder.config.layer_norm_eps,
            batch_first=True).encoder
        
        # Pooling layer.
        self.pooler = Pooling(pooling_mode)

    def forward(self, token_ids: T, attention_masks: T, input_ids: Optional[T] = None) -> Tuple[T, T, T]:
        """
        Args:
            token_ids: 3D tensor of size [batch_size, batch_max_chunks, max_chunk_length].
            attention masks: 3D tensor of size [batch_size, batch_max_chunks, max_chunk_length].
            input_ids: 1D tensor of size [batch_size]. Optional: unique IDs of input texts.
        Returns:
            text_representations: 2D tensor of size [batch_size, embedding_dim].
        """
        # Reshape to 2D tensor of size [batch_size x batch_max_chunks, max_chunk_length].
        token_ids_reshaped = token_ids.contiguous().view(-1, token_ids.size(-1))
        attention_masks_reshaped = attention_masks.contiguous().view(-1, attention_masks.size(-1))

        # Compute context-aware word embeddings with the BERT-based encoder.
        token_embeddings = self.word_encoder(input_ids=token_ids_reshaped, attention_mask=attention_masks_reshaped, return_dict=False)[0]

        # Reshape back to 4D tensor of size [batch_size, batch_max_chunks, max_chunk_length, hidden_size]
        token_embeddings = token_embeddings.contiguous().view(*tuple(token_ids.size()), self.dim)

        # Keep only [CLS] embeddings (and corresponding attention masks) for each chunk in the batch.
        chunk_embeddings = token_embeddings[:,:,0,:] #-> 3D tensor of size [batch_size, batch_max_chunks, hidden_size]
        chunk_masks = attention_masks[:,:,0] #-> 2D tensor of size [batch_size, batch_max_chunks]

        # Compute chunk positional embeddings and add them to the chunk embeddings.
        chunk_positions = torch.tensor(range(1, token_ids.size(1)+1), dtype=torch.int, device=token_ids.device) * chunk_masks.int()
        chunk_embeddings += self.chunk_pos_embeddings(chunk_positions)

        # Compute context-aware chunk embeddings with the Transformer-based encoder.
        chunk_embeddings = self.chunk_encoder(chunk_embeddings)

        # Pool the context-aware chunk embeddings to distill a global representation for each document in the batch.
        doc_embeddings = self.pooler(chunk_embeddings, chunk_masks)
        return doc_embeddings

    @staticmethod
    def sinusoidal_init(num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        """https://github.com/pytorch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py#L36
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1) #zero pad
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
