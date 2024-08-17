import torch
import torch.nn as nn
import math

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Check and correct the shape of the masks
        if tgt_mask is not None:
            if tgt_mask.dim() == 2:
                tgt_mask = tgt_mask.unsqueeze(0)
            elif tgt_mask.dim() == 3:
                tgt_mask = tgt_mask.unsqueeze(1).repeat(tgt.shape[0], 1, 1, 1)
            elif tgt_mask.dim() == 4:
                tgt_mask = tgt_mask.squeeze(2)
            print(f"tgt_mask shape after processing: {tgt_mask.shape}")

        if memory_mask is not None:
            if memory_mask.dim() == 2:
                memory_mask = memory_mask.unsqueeze(1)
            elif memory_mask.dim() == 3:
                memory_mask = memory_mask.unsqueeze(1)
            elif memory_mask.dim() == 4:
                memory_mask = memory_mask.squeeze(1)
            print(f"memory_mask shape after processing: {memory_mask.shape}")

        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return self.norm(output)