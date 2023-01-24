import math
import numpy as np
from typing import Optional, List, Union, Tuple

import torch
from torch import nn, Tensor as T
from transformers import AutoTokenizer


class LongTokenizer:
    def __init__(self, 
                 tokenizer_name_or_path: str,
                 max_document_length: Optional[int] = None, #length to which the input document will be truncated before tokenization. Default to None (no truncation).
                 max_chunk_length: Optional[int] = None, #maximum length of the chunks. Default to None (model max length).
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=True)
        self.max_document_length = float('inf') if max_document_length is None else max_document_length
        self.max_chunk_length = self.tokenizer.model_max_length if max_chunk_length is None else max_chunk_length


    def __call__(self, texts: Union[str, List[str]]) -> Tuple[T, T]:
        """
        Args:
            texts: a list of (short or long) texts to tokenize.
        Returns:
            token_ids: 3D tensor of size [batch_size, max_chunks, max_chunk_length].
            attention masks: 3D tensor of size [batch_size, max_chunks, max_chunk_length].
        """
        # Tokenize inputs texts: (i) without truncating if max_document_length is inf; (ii) by truncating to max_document_length otherwise. 
        # NB: not truncating long texts might raise an OOM error during training if too little GPU memory.
        if np.isinf(self.max_document_length):
            tokenized = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        else:
            tokenized = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_document_length, return_tensors="pt")
        
        # Split long tokenized texts into smaller chunks (only if max_chunk_length is smaller than max_doc_length).
        if self.max_document_length > self.max_chunk_length:
            return self.__chunk_tokenized_inputs(tokenized['input_ids'], tokenized['attention_mask'])
        else:
            return tokenized['input_ids'], tokenized['attention_mask']


    def __chunk_tokenized_inputs(self, token_ids: T, attention_masks: T) -> Tuple[T,T]:
        """Chunk the tokenized inputs returned by HuggingFace tokenizer into fixed-lengths units.
        Args:
            token_ids: 2D tensor of size [batch_size, batch_max_seq_len].
            attention_masks: 2D tensor of size [batch_size, batch_max_seq_len].
        Returns:
            token_ids: 3D tensor of size [batch_size, batch_max_chunks, max_chunk_length].
            attention_masks: 3D tensor of size [batch_size, batch_max_chunks, max_chunk_length].
        """
        batch_size, batch_max_seq_len = token_ids.shape
        window_size = self.max_chunk_length // 10 #sliding window with 10% overlap.

        if batch_max_seq_len <= self.max_chunk_length:
            # If the max sequence length from the current batch is smaller than the defined chunk size, simply return the tensors with a dim of 1 on the 'batch_max_chunks' dimension.
            return token_ids.unsqueeze(1), attention_masks.unsqueeze(1)
        else:
            # Remove first column from 2D tensor (corresponding to the CLS tokens of the long sequences).
            token_ids = token_ids[:, 1:] #T[batch_size, batch_max_seq_len-1]
            attention_masks = attention_masks[:, 1:] #T[batch_size, batch_max_seq_len-1]
            batch_max_seq_len -= 1
            max_chunk_length = self.max_chunk_length - 1

            # Pad 2D tensor so that the 'batch_seq_len' is a multiple of 'max_chunk_length' (otherwise unfold ignore remaining tokens).
            num_windows = math.floor((batch_max_seq_len - max_chunk_length)/(max_chunk_length - window_size))
            num_repeated_tokens = num_windows * window_size
            batch_seq_len = math.ceil((batch_max_seq_len + num_repeated_tokens)/max_chunk_length) * max_chunk_length
            token_ids = nn.functional.pad(input=token_ids, pad=(0, batch_seq_len - batch_max_seq_len), mode='constant', value=self.tokenizer.pad_token_id)
            attention_masks = nn.functional.pad(input=attention_masks, pad=(0, batch_seq_len - batch_max_seq_len), mode='constant', value=0)

            # Split tensor along y-axis (i.e., along the 'batch_max_seq_len' dimension) with overlapping of 'window_size'
            # and create a new 3D tensor of size [batch_size, max_chunk_length-1, batch_max_seq_len/max_chunk_length].
            token_ids = token_ids.unfold(dimension=1, size=max_chunk_length, step=max_chunk_length-window_size)
            attention_masks = attention_masks.unfold(dimension=1, size=max_chunk_length, step=max_chunk_length-window_size)
            batch_max_chunks = token_ids.size(1)

            # Append CLS token id before each chunk if the latter does not start with a PAD token. If so, append the PAD token id instead.
            cls_token_ids = torch.full((batch_size, batch_max_chunks, 1), self.tokenizer.cls_token_id)
            cls_token_ids[token_ids[:,:,0].unsqueeze(2) == self.tokenizer.pad_token_id] = self.tokenizer.pad_token_id
            token_ids = torch.cat([cls_token_ids, token_ids], axis=2)

            # Add attention masks of 1 for the new CLS tokens, and masks of 0 for the new PAD tokens.
            cls_attention_masks = torch.ones(batch_size, batch_max_chunks, 1)
            cls_attention_masks[cls_token_ids == self.tokenizer.pad_token_id] = 0
            attention_masks = torch.cat([cls_attention_masks, attention_masks], axis=2)

            # Return tokens and masks.
            return token_ids, attention_masks
