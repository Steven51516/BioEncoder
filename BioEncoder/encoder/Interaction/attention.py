import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionInteract(nn.Module):
    """
        多头注意力的交互层
    """

    def __init__(self, embed_size, head_num, dropout, residual=True):
        """
        """
        super(MultiHeadAttentionInteract, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, head_num, 0.0)

    def forward(self, x, y):
        """
            x : (batch_size, feature_fields, embed_dim)
        """

        y = y.unsqueeze(1)  # Shape becomes (batch_size, 1, embed_dim)
        results1 = torch.cat([x, y], dim=1)  # Shape becomes (batch_size, feature_fields+1, embed_dim)
        x = results1
        x_transposed = x.transpose(0, 1)

        attn_output, attn_output_weights = self.attention(x_transposed, x_transposed, x_transposed)

        # Transpose attn_output back to original shape: (batch_size, seq_len, embed_dim)
        results = attn_output.transpose(0, 1)


        #modified
        summed_x = torch.sum(results, dim=1)  # Exclude the portion corresponding to y

        # Concatenate summed_x with y
        results = torch.cat([summed_x, y.squeeze(1)], dim=1)  # Shape: (batch_size, 2*embed_dim)
        # print(results)
        return results