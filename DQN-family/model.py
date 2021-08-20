import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd 

def Model(inp_dim,out_dim,cfg):
    if cfg.model == 'MLP':
        return MLP(inp_dim,out_dim,cfg)
    elif cfg.model == 'CNN':
        return CNN(inp_dim,out_dim,cfg)
    elif cfg.model == 'DuelingCNN':
        return DuelingCNN(inp_dim,out_dim,cfg)
    elif cfg.model == 'NoisyCNN':
        return NoisyCNN(inp_dim,out_dim,cfg)
    elif cfg.model == 'CategoricalCNN':
        num_actions = out_dim
        return CategoricalCNN(inp_dim, num_actions, cfg.num_atoms, cfg)
    elif cfg.model == 'RainbowCNN':
        num_actions = out_dim
        return RainbowCNN(inp_dim, num_actions, cfg.num_atoms, cfg)


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
        x = self.fc3(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, inp_dim, out_dim, cfg):
        super(CNN, self).__init__()
        
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        
        self.features = nn.Sequential(
            nn.Conv2d(inp_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, self.out_dim)
        )
        
    def forward(self, x):
        x = self.features(x)
#         x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        with torch.no_grad():
#         return self.features(autograd.Variable(torch.zeros(1, *self.inp_dim))).reshape(1, -1).shape[1]
            return self.features(torch.zeros(1, *self.inp_dim)).reshape(1, -1).shape[1]

class DuelingCNN(CNN):
    def __init__(self, inp_dim, out_dim, cfg):
        super(DuelingCNN, self).__init__(inp_dim, out_dim, cfg)
        
        self.advantage = self.fc
        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()
    
class NoisyCNN(CNN):
    def __init__(self, inp_dim, out_dim, cfg):
        super(NoisyCNN, self).__init__(inp_dim, out_dim, cfg)
        self.noisy1 = NoisyLinear(self.feature_size(),cfg.hidden_dim,cfg.noisy_std)
        self.noisy2 = NoisyLinear(cfg.hidden_dim,self.out_dim,cfg.noisy_std)
    def forward(self, x):
#         x = x / 255.
        x = self.features(x)
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

class CategoricalCNN(CNN):
    def __init__(self, inp_dim, num_actions, num_atoms, cfg):
        super(CategoricalCNN, self).__init__(inp_dim, num_actions*num_atoms, cfg)
        self.Vmin = cfg.Vmin
        self.Vmax = cfg.Vmax
        self.num_atoms = num_atoms
        self.num_actions = num_actions
    def forward(self,x):
#         x = x / 255.
        x = self.features(x)
        x = self.fc(x)
        x =  F.softmax(x.reshape(-1,self.num_atoms),dim=1).reshape(-1,self.num_actions,self.num_atoms) #(b,nA*nC) -> (b,nA,nC)
#         x = F.softmax(x.reshape(-1,self.num_actions,self.num_atoms),dim=2)
        return x
    
class RainbowCNN(nn.Module):
    def __init__(self, inp_dim, num_actions, num_atoms, cfg):
        super(RainbowCNN, self).__init__()
        self.inp_dim = inp_dim
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = cfg.Vmin
        self.Vmax = cfg.Vmax
        
        self.features = nn.Sequential(
            nn.Conv2d(inp_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.noisy_value1 = NoisyLinear(self.feature_size(),cfg.hidden_dim,cfg.noisy_std)
        self.noisy_value2 = NoisyLinear(cfg.hidden_dim,self.num_atoms,cfg.noisy_std)
        self.noisy_advantage1 = NoisyLinear(self.feature_size(),cfg.hidden_dim,cfg.noisy_std)
        self.noisy_advantage2 = NoisyLinear(cfg.hidden_dim,self.num_actions*self.num_atoms,cfg.noisy_std)
    
    def forward(self, x):
#         x = x / 255.
        x = self.features(x)
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value).reshape(-1,1,self.num_atoms) #(b,nC) -> (b,1,nC)
        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage).reshape(-1,self.num_actions,self.num_atoms) #(b,nA*nC) -> (b,nA,nC)
        
        x = value + advantage - advantage.mean(1,keepdim=True) # (b,nA,nC)
        x =  F.softmax(x.reshape(-1,self.num_atoms),dim=1).reshape(-1,self.num_actions,self.num_atoms) #(b,nA*nC) -> (b,nA,nC)
        return x
    
    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()
    
    def feature_size(self):
        with torch.no_grad():
            return self.features(torch.zeros(1, *self.inp_dim)).reshape(1, -1).shape[1]

# +
# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

#     def _scale_noise(self, size):
#         x = torch.randn(size, device=self.weight_mu.device)
#         return x.sign().mul_(x.abs().sqrt_())
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)
# -


