import torch
import numpy as np
from tqdm import tqdm
from typing import List, Type, Union, Optional

from src.data.long_tokenizer import LongTokenizer



def encode(model: Type[torch.nn.Module], #
           tokenizer: Type[LongTokenizer], #
           texts: Union[str, List[str]], #
           batch_size: int, #
           text_ids: Optional[Union[str, List[str]]] = None, #
           show_progress: bool = True, #
           device: str = 'cuda' if torch.cuda.is_available() else 'cpu', #
    ):
    if isinstance(texts, str):
        texts = [texts]
    if text_ids is not None:
        if isinstance(text_ids, str):
            text_ids = [text_ids]
        assert len(texts) == len(text_ids), "Length of 'text_ids' doesn't match length of 'texts'."
    
    # Sort texts by length to get batches of similar lengths.
    length_sorted_idx = np.argsort([-len(t) for t in texts])
    texts_sorted = [texts[idx] for idx in length_sorted_idx]
    if text_ids is not None:
        ids_sorted = [text_ids[idx] for idx in length_sorted_idx]

    model.eval()
    model.to(device)
    all_embeddings = []
    for start_idx in tqdm(range(0, len(texts), batch_size), desc=f"- Encoding batches of {batch_size} docs", disable=not show_progress, leave=False):
        if text_ids is not None:
            # Get ids of batch of documents.
            ids_batch = ids_sorted[start_idx:start_idx+batch_size]
            ids_batch = torch.tensor(ids_batch, dtype=torch.int)

        # Tokenize batch of long documents.
        texts_batch = texts_sorted[start_idx:start_idx+batch_size]
        token_ids, attention_masks = tokenizer(texts_batch)

        # Send tensors to device.
        token_ids = token_ids.to(device)
        attention_masks = attention_masks.to(device)
        if text_ids is not None:
            ids_batch = ids_batch.to(device)

        # Encode.
        with torch.no_grad():
            if text_ids is not None:
                embeddings = model(token_ids, attention_masks, ids_batch)
            else:
                embeddings = model(token_ids, attention_masks)
            all_embeddings.extend(embeddings)

    # Sort the embeddings back in the original order of the input docs and returns torch tensor.
    all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
    return torch.stack(all_embeddings).detach().cpu()
