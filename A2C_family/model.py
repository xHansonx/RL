# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd 
from torch.distributions import Normal, Categorical


def Model(inp_dim,act_dim,cfg):
    if cfg.model == 'Discrete_MLP':
        return Discrete_MLP(inp_dim,act_dim,cfg.hidden_dim)
    elif cfg.model == 'Discrete_CNN':
        return Discrete_CNN(inp_dim,act_dim,cfg.hidden_dim)
    elif cfg.model == 'Continuous_MLP':
        return Continuous_MLP(inp_dim,act_dim,cfg.hidden_dim,cfg.max_act)
#     elif cfg.model == 'Continuous_CNN':
#         return Continuous_CNN(inp_dim,act_dim,cfg.hidden_dim)


# +
class Actor_MLP(nn.Module):
    def __init__(self, inp_dim, act_dim, hidden_dim=256, max_act=1.0):
        super(Actor_MLP, self).__init__()
        self.inp_dim = inp_dim
        self.max_act = max_act
        
        self.features = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.ReLU())
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh())

    def forward(self, state):
        x = self.features(state)
        x = self.max_act * self.actor(x)
        return x
        
class Critic_MLP(nn.Module): 
    def __init__(self, inp_dim, act_dim, hidden_dim=256):
        super(Critic_MLP, self).__init__()
        self.inp_dim = inp_dim
        
        # assume input has same dimension with action, and
        # inp_dim, act_dim, inp_dim+act_dim << hidden_dim
        self.features = nn.Sequential(
            nn.Linear(inp_dim+act_dim, hidden_dim),
            nn.ReLU())
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))
    
    def forward(self, state, action):
        x = self.features(torch.cat([state,action],dim=-1))
        x = self.critic(x)
        return x
# -



# +
class Discrete_MLP(nn.Module):
    def __init__(self, inp_dim, act_dim, hidden_dim=256):
        super(Discrete_MLP, self).__init__()
        self.inp_dim = inp_dim
        
        self.features = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.ReLU())
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=1))
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = self.features(x)
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return value, dist
    
#     def feature_size(self):
#         with torch.no_grad():
#             return self.features(torch.zeros(1, *self.inp_dim)).reshape(1, -1).shape[1]


# -

class Discrete_CNN(nn.Module):
    def __init__(self, inp_dim, act_dim, hidden_dim=512):
        super(Discrete_CNN, self).__init__()
        self.inp_dim = inp_dim

        self.features = nn.Sequential(
            nn.Conv2d(inp_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten())
        
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size(), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=1))
        
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size(), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x):
#         x = x/255. 
        x = self.features(x)
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return value, dist
    
    def feature_size(self):
        with torch.no_grad():
            return self.features(torch.zeros(1, *self.inp_dim)).reshape(1, -1).shape[1]


# +
class Continuous_MLP(nn.Module):
    def __init__(self, inp_dim, act_dim, hidden_dim=256, max_act=1.0):
        super(Continuous_MLP, self).__init__()
        self.inp_dim = inp_dim
        self.max_act = max_act
        
        self.features = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.ReLU())
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim))
        self.fc = nn.Linear(act_dim, act_dim)
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, x):
        x = self.features(x)
        actor = self.actor(x)
        value = self.critic(x)
        
        actor_mu = self.max_act*torch.tanh(actor)
        actor_sigma = F.softplus(self.fc(actor))
        actor_sigma = torch.clamp(actor_sigma, 1e-9, 3)
        dist = Normal(actor_mu, actor_sigma)
        return value, dist

# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, mean=0., std=0.1)
#         nn.init.constant_(m.bias, 0.1)

# class Continuous_MLP(nn.Module):
#     def __init__(self, num_inputs, num_outputs, hidden_size=256, std=0.1):
#         super(Continuous_MLP, self).__init__()
        
#         self.critic = nn.Sequential(
#             nn.Linear(num_inputs, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )
        
#         self.actor = nn.Sequential(
#             nn.Linear(num_inputs, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, num_outputs),
#         )
#         self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
#         self.apply(init_weights)
        
#     def forward(self, x):
#         value = self.critic(x)
#         mu    = self.actor(x)
# #         std   = self.log_std.exp().expand_as(mu)
#         std = torch.clamp(self.log_std.exp(), 1e-3, 50)
#         dist  = Normal(mu, std)
#         return value, dist

# +
# class Continuous_CNN(nn.Module):
#     def __init__(self, inp_dim, act_dim, hidden_dim=512):
#         super(Continuous_CNN, self).__init__()
#         self.inp_dim = inp_dim
        
#         self.features = nn.Sequential(
#             nn.Conv2d(inp_dim[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten())
        
#         self.actor = nn.Sequential(
#             nn.Linear(self.feature_size(), hidden_dim),
#             nn.ReLU())
#         self.fc = nn.Linear(hidden_dim, act_dim)
        
#         self.critic = nn.Sequential(
#             nn.Linear(self.feature_size(), hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1))

#     def forward(self, x):
#         x = self.features(x)
#         x_actor = self.actor(x)
#         value = self.critic(x)
        
#         actor_mu = torch.tanh(self.fc(x_actor))
#         actor_sigma = F.softplus(self.fc(x_actor))
#         dist = Normal(actor_mu, actor_sigma)
#         return value, dist
    
#     def feature_size(self):
#         with torch.no_grad():
#             return self.features(torch.zeros(1, *self.inp_dim)).reshape(1, -1).shape[1]

# +
# import os,sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# use_cuda = torch.cuda.is_available()
# device   = torch.device("cuda" if use_cuda else "cpu")

# # model = Discrete_CNN((1,84,84),6).to(device)
# model = Continuous_MLP(3,1).to(device)

# from torchsummary import summary
# # summary(model,(1,84,84))
# summary(model,(3,))

# +
# model = Continous_CNN((1,84,84),6).to(device)

# from torchsummary import summary
# summary(model,(1,84,84))

# +
# value,dist = model(torch.randn(1,1,84,84,device=device))
# +
# value
# dist.sample()
# +
# x = torch.rand(4,6)
# x = nn.Linear(6,2)(x)
# x = torch.tanh(x)
# x.shape #[b,nA]
# -


