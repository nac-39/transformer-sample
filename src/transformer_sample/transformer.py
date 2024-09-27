import torch.nn as nn
from transformer_sample.encoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=100,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)

        if self.training:  # トレーニング中のみ表示ï
            from torchviz import make_dot

            make_dot(enc_src, params=dict(self.named_parameters())).render(
                "graph", format="png"
            )
        return enc_src, trg
