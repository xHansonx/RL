# -*- coding: utf-8 -*-
import numpy as np
import torch 
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from model import Model


# +
class Reinforce(object):
    def __init__(self,state_dim,action_dim,cfg):
        self.state_dim = state_dim    # dimension of state
        self.action_dim = action_dim  # dimension of action
        self.lr = cfg.lr              # learning rate
        self.gamma = cfg.gamma        # discount rate of reward
#         self.epsilon = 0 
#         self.sample_count = 0  
#         self.epsilon_start = cfg.epsilon_start
#         self.epsilon_end = cfg.epsilon_end
#         self.epsilon_decay = cfg.epsilon_decay
        self.log_action_probs = []
        self.eps = np.finfo(np.float32).eps.item()
        self.policy_net = Model(state_dim,action_dim,cfg)
#         self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=cfg.lr)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
    def sample(self,state):
#         self.sample_count += 1
#         self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
#             np.exp(-1. * self.sample_count / self.epsilon_decay)
#         if np.random.uniform(0, 1) > self.epsilon:  # 随机选取0-1之间的值，如果大于epsilon就按照贪心策略选取action，否则随机选取
#             state = torch.from_numpy(state)#.float()
#             action = Categorical(self.policy_net(state)).sample().item()
#         else:
#             action = np.random.choice(self.action_dim)  #有一定概率随机探索选取一个动作
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = Categorical(self.policy_net(state))
        action = probs.sample()
        self.log_action_probs.append(probs.log_prob(action))
        return action.item()
    def predict(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = torch.argmax(self.policy_net(state)).item()
        return action
    def learn(self,batch_state,batch_action,batch_reward):
        R = 0
        Gt = []
        for r in batch_reward[::-1]:
            if r == 0:
                R = r
            else:
                R = r + self.gamma*R
            Gt.append(R)
        Gt = Gt[::-1]
        
#         batch_reward = batch_reward[::-1]
#         for i in range(len(batch_reward)-1):
#             prev_r,r = batch_reward[i+1],batch_reward[i]
#             if r == 0:
#                 R = r
#             elif 
#             else:
#                 R = r + self.gamma*R
#             Gt.append(R)
#         Gt.append(batch_reward[-1] + self.gamma*Gt[-1])
        
        relative_advantage = torch.tensor(Gt)
        relative_advantage = (relative_advantage - relative_advantage.mean())/(relative_advantage.std()+self.eps)
        
#         loss = [(-A*log_prob).unsqueeze(0) for A,log_prob in zip(relative_advantage,batch_log_probs)]
        loss = [(-A*log_prob) for A,log_prob in zip(relative_advantage,self.log_action_probs)]
        total_loss = torch.cat(loss).sum()
        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(total_loss, 0.1)
        self.optimizer.step()
        del self.log_action_probs[:]
        
    def save(self,path):
        torch.save(self.policy_net.state_dict(), path+'pg_checkpoint.pt')
    def load(self,path):
        self.policy_net.load_state_dict(torch.load(path+'pg_checkpoint.pt')) 


# -
class Reinforce_baseline(Reinforce):
    def __init__(self,state_dim,action_dim,cfg):
        super(Reinforce_baseline,self).__init__(state_dim,action_dim,cfg)
        self.value_net =  Model(state_dim,1,cfg)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=cfg.lr)
        self.values = []
    def sample(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = Categorical(self.policy_net(state))
        action = probs.sample()
        self.log_action_probs.append(probs.log_prob(action))
        self.values.append(self.value_net(state))
        return action.item()
    def learn(self,batch_state,batch_action,batch_reward):
        R = 0
        Gt = []
        for r in batch_reward[::-1]:
            if r == 0:
                R = r
            else:
                R = r + self.gamma*R
            Gt.append(R)
        Gt = Gt[::-1]
        relative_advantage = torch.tensor(Gt) - torch.cat(self.values,1)
        loss_value = -relative_advantage.sum(dim=1)
        loss_policy = -relative_advantage * torch.cat(self.log_action_probs)
        loss_policy = loss_policy.sum(dim=1)

        self.value_optimizer.zero_grad()
        self.optimizer.zero_grad()
        loss_value.backward(retain_graph = True)
        loss_policy.backward(retain_graph = True)
        clip_grad_norm_(loss_value, 0.1)
        clip_grad_norm_(loss_policy, 0.1)
        self.value_optimizer.step()
        self.optimizer.step()
        
        del self.values[:]
        del self.log_action_probs[:]



def Agent(state_dim,action_dim,cfg):
    if cfg.agent == 'Reinforce':
        return Reinforce(state_dim,action_dim,cfg)
    elif cfg.agent == 'Reinforce_baseline':
        return Reinforce_baseline(state_dim,action_dim,cfg)




