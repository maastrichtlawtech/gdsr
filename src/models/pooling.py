import torch
from torch import nn, Tensor as T


class Pooling(nn.Module):
    """Performs pooling (mean or max) on the token embeddings. 
    Using pooling, it generates from a variable sized text passage a fixed sized passage embedding.
    """
    def __init__(self, pooling_mode: str):
        super(Pooling, self).__init__()
        assert pooling_mode in ['mean', 'max'], f"ERROR: Unknown pooling strategy '{pooling_mode}'"
        self.pooling_mode = pooling_mode

    def forward(self, token_embeddings: T, attention_masks: T) -> T:
        """
        Args:
            token_embeddings: 3D tensor of size [batch_size, seq_len, embedding_dim].
            attention_masks: 2D tensor of size [batch_size, seq_len].
        Returns:
            text_vectors: 2D tensor of size [batch_size, embedding_dim].
        """
        if self.pooling_mode == 'max':
            # Set all values of the [PAD] embeddings to large negative values (so that they are never considered as maximum for a channel).
            attention_masks_expanded = attention_masks.unsqueeze(-1).expand(token_embeddings.size())
            token_embeddings[attention_masks_expanded == 0] = -1e+9 if token_embeddings.dtype == torch.float32 else -1e+4
            # Compute the maxima along the 'seq_length' dimension (-> Tensor[batch_size, embedding_dim]).
            text_vectors = torch.max(token_embeddings, dim=1).values
        else:
            # Set all values of the [PAD] embeddings to zeros (so that they are not taken into account in the sum for a channel).
            attention_masks_expanded = attention_masks.unsqueeze(-1).expand(token_embeddings.size())
            token_embeddings[attention_masks_expanded == 0] = 0.0
            # Compute the means by first summing along the 'seq_length' dimension (-> Tensor[batch_size, embedding_dim]).
            sum_embeddings = torch.sum(token_embeddings, dim=1)
            # Then, divide all values of a passage vector by the original passage length.
            sum_mask = attention_masks_expanded.sum(dim=1) # -> Tensor[batch_size, embedding_dim] where each value is the length of the corresponding passage.
            sum_mask = torch.clamp(sum_mask, min=1e-7) # Make sure not to have zeros by lower bounding all elements to 1e-7.
            text_vectors = sum_embeddings / sum_mask # Divide each dimension by the sequence length.
        return text_vectors
