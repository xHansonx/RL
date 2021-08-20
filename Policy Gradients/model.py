# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


# +
class MLP(nn.Module):
    def __init__(self,inp_dim,out_dim,cfg):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(inp_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim,cfg.hidden_dim)
        self.fc3 = nn.Linear(cfg.hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
#         x = nn.Dropout(p=0.6)(x)
        x = F.relu(self.fc2(x))
#         x = nn.Dropout(p=0.6)(x)
        x = F.softmax(self.fc3(x))
        return x
    
def Model(inp_dim,out_dim,cfg):
    if cfg.model == 'MLP':
        return MLP(inp_dim,out_dim,cfg)
# -

