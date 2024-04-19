import math
import torch
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from layers.AMS import AMS
from layers.Layer import WeightGenerator, CustomLinear
from layers.RevIN import RevIN
from functools import reduce
from operator import mul


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer_nums = configs.layer_nums  # 设置pathway的层数
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.num_experts = configs.num_experts
        self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.gpu))

        self.AMS = AMS(self.seq_len, self.seq_len, self.num_experts, self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list, noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=1, residual_connection=self.residual_connection)
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )
        self.experts_projections = nn.ModuleList()
        for i in range(self.num_experts):
            self.experts_projections.append(nn.Sequential(nn.Linear(self.seq_len * self.d_model, self.pre_len)))

    def forward(self, x):

        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = self.start_fc(x.unsqueeze(-1))


        batch_size = x.shape[0]

        out, expert_outputs = self.AMS(out)

        out = out.permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        for i in range(self.num_experts):
            expert_outputs[i] = expert_outputs[i].permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)
            expert_outputs[i] = self.experts_projections[i](expert_outputs[i]).transpose(2, 1)

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')
        for i in range(self.num_experts):
                expert_outputs[i] = self.revin_layer(expert_outputs[i], 'denorm')

        return out, expert_outputs


