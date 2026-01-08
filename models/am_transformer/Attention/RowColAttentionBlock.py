import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class RowColAttentionBlock(nn.Module):
    """Improved Row and Column attention block for tabular data_utils."""

    def __init__(
            self,
            dim,
            heads,
            attn_dropout=0.1,
            feature_groups=1,
            use_cls_token=True,
            qk_relu=False
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.feature_groups = feature_groups
        self.use_cls_token = use_cls_token
        self.qk_relu = qk_relu
        self.scale = dim ** -0.5

        # Pre LayerNorms for row and column attention
        self.row_prenorm = nn.LayerNorm(dim)
        self.col_prenorm = nn.LayerNorm(dim)

        # Row attention components
        self.row_to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.row_to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(attn_dropout)
        )

        # Column attention components
        self.col_to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.col_to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(attn_dropout)
        )

        # Learnable weights for combining row and column attention
        self.row_weight = nn.Parameter(torch.ones(1))
        self.col_weight = nn.Parameter(torch.ones(1))

        # Output components
        self.final_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(attn_dropout)

    def _split_heads(self, x):
        """Split tensor into attention heads with improved shape handling."""
        b, n, d = x.shape
        head_dim = d // self.heads
        x = x.reshape(b, n, self.heads, head_dim)
        return x.transpose(1, 2)  # (b, h, n, d_head)

    def _merge_heads(self, x):
        """Merge attention heads with improved shape handling."""
        b, h, n, d = x.shape
        x = x.transpose(1, 2).contiguous()  # (b, n, h, d)
        return x.reshape(b, n, h * d)

    def _attention(self, q, k, v, mask=None):
        """Compute attention with improved numerical stability."""
        # Scale dot-product
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply ReLU if specified
        if self.qk_relu:
            dots = F.relu(dots)

        # Apply mask if provided
        if mask is not None:
            dots = dots.masked_fill(mask == 0, float('-inf'))

        # Attention weights with improved numerical stability
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        return out

    def forward(self, x, mask=None):
        """Forward pass with improved attention computation."""
        b, n, d = x.shape
        cls_offset = 1 if self.use_cls_token else 0

        # Row attention
        x_row = self.row_prenorm(x)
        row_qkv = self.row_to_qkv(x_row).chunk(3, dim=-1)
        row_q, row_k, row_v = map(self._split_heads, row_qkv)

        row_out = self._attention(row_q, row_k, row_v, mask)
        row_out = self._merge_heads(row_out)
        row_out = self.row_to_out(row_out)

        # Column attention
        x_col = x[:, cls_offset:]
        x_col = einops.rearrange(
            x_col,
            'b (g f) d -> (b g) f d',
            g=self.feature_groups
        )
        x_col = self.col_prenorm(x_col)

        col_qkv = self.col_to_qkv(x_col).chunk(3, dim=-1)
        col_q, col_k, col_v = map(self._split_heads, col_qkv)

        col_out = self._attention(col_q, col_k, col_v)
        col_out = self._merge_heads(col_out)
        col_out = self.col_to_out(col_out)

        # Reshape column HS数据集相关性分析
        col_out = einops.rearrange(
            col_out,
            '(b g) f d -> b (g f) d',
            g=self.feature_groups
        )

        # Add CLS token back to column HS数据集相关性分析
        if self.use_cls_token:
            col_out = torch.cat([x[:, :1], col_out], dim=1)

        # Weighted combination of row and column attention
        out = self.row_weight * row_out + self.col_weight * col_out

        # Final normalization and residual
        out = self.final_norm(out)
        out = out + x  # Residual connection

        return out