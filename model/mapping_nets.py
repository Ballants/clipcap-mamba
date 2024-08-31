from copy import deepcopy

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm


class MambaMapping(nn.Module):
    def __init__(self, d_clip, embed_size, prefix_len, num_mamba_layers, device):
        super(MambaMapping, self).__init__()
        self.d_clip = d_clip
        self.prefix_len = prefix_len
        self.embed_size = embed_size
        self.num_mamba_layers = num_mamba_layers
        self.device = device

        self.fc = nn.Linear(self.d_clip, self.prefix_len * self.embed_size)

        # Each mamba bloch uses roughly 3 * expand * d_clip^2 parameters
        self.mamba = nn.ModuleList([deepcopy(Mamba(d_model=self.embed_size, d_state=16, d_conv=4, expand=2)) for _ in
                                    range(self.num_mamba_layers)])
        factory_kwargs = {"device": self.device, "dtype": torch.float32}
        self.norm_f = RMSNorm(self.embed_size, eps=1e-05, **factory_kwargs)

        self._initialize_weights()

    def forward(self, src):
        bs = src.shape[0]  # batch_size

        src = self.fc(src)
        src = src.view(bs, self.prefix_len, self.embed_size)  # batch_size, prefix_len * embed_size
        src = nn.BatchNorm1d(src.size(1))(src)
        src = nn.SiLU()(src)
        src = nn.Dropout(p=0.5)(src)
        src = src.view(bs, self.prefix_len, self.embed_size)  # batch_size, prefix_len, embed_size

        for layer in self.mamba:
            src = layer(src)  # batch_size, prefix_len, embed_size

        out = self.norm_f(src)  # batch_size, prefix_len, embed_size

        return out

    def _initialize_weights(self):

        nn.init.xavier_uniform_(self.fc.weight)

        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
