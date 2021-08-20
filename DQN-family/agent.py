# -*- coding: utf-8 -*-
import numpy as np
import torch 
# from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from model import Model
from memory import ReplayBuffer

def Agent(state_dim,action_dim,cfg):
    if cfg.agent == 'DQN':
        return DQN(state_dim,action_dim,cfg)
    elif cfg.agent == 'double_DQN':
        return double_DQN(state_dim,action_dim,cfg)
    elif cfg.agent == 'dueling_DQN':
        return dueling_DQN(state_dim,action_dim,cfg)
    elif cfg.agent == 'prioritized_DQN':
        return prioritized_DQN(state_dim,action_dim,cfg)
    elif cfg.agent == 'noisy_DQN':
        return noisy_DQN(state_dim,action_dim,cfg)
    elif cfg.agent == 'ctegorical_DQN':
        return ctegorical_DQN(state_dim,action_dim,cfg)
    elif cfg.agent == 'rainbow_DQN':
        return rainbow_DQN(state_dim,action_dim,cfg)


class DQN(object):
    def __init__(self,state_dim,action_dim,cfg):
        self.batch_size = cfg.batch_size              # number of examples in one batch
        self.n_multi_step = cfg.n_multi_step          # number of steps of multi-step style gain
        self.gamma = cfg.gamma                        # discount rate of reward
        self.lr = cfg.lr                              # learning rate
        self.state_dim = state_dim                    # dimension of state
        self.action_dim = action_dim                  # dimension of action
        self.sync_target_freq = cfg.sync_target_freq  # number of learning steps for sync target model
        self.sample_steps = 0                         # number of steps of sample()
        self.learning_steps = 0                       # number of steps of learning
        
        self.device = cfg.device                      # cpu or gpu
        self.epsilon =  lambda sample_steps: cfg.epsilon_end + \
                                            (cfg.epsilon_start - cfg.epsilon_end) * \
                                            np.exp(-0.3 * sample_steps * cfg.epsilon_decay)

        self.policy_net = Model(state_dim,action_dim,cfg).to(self.device)
        self.target_net = Model(state_dim,action_dim,cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        ## copy params from policy net to target net ##
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): 
            target_param.data.copy_(param.data)
#         self.memory = ReplayBuffer(cfg.memory_size)
            
    def sample(self,state):
#         if np.random.rand() < self.epsilon(self.sample_steps):
#             action = np.random.choice(self.action_dim)
#         else:
#             action = self.predict(state)
#         self.sample_steps += 1
#         return action
        with torch.no_grad():
            action = self.predict(state)
            mask = torch.rand(action.shape,device=self.device) < self.epsilon(self.sample_steps)
            rand_action = torch.randint_like(action,self.action_dim)
            action = torch.where(mask,rand_action,action)
            self.sample_steps += 1
#         return actions.cpu().numpy()
        return action.cpu().numpy() if action.shape[0] > 1 else action.item()
    
    def predict(self,state):
        with torch.no_grad():
            if len(state.shape) == 4:
                state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            else:
                state = torch.as_tensor([state], device=self.device, dtype=torch.float32)
    #         action = torch.argmax(self.policy_net(state)).item()
            action = self.policy_net(state).max(1)[1]#.item()
        return action    
    
    def learn(self,batch_state,batch_action,batch_reward,batch_next_state,batch_done):
        if self.learning_steps % self.sync_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learning_steps += 1
#         batch_state,batch_action,batch_reward,batch_next_state,batch_done = self.memory.sample(self.batch_size)
        batch_state = torch.as_tensor(batch_state, device=self.device, dtype=torch.float32) #(b,c,?,?) or (b,nS)
        batch_action = torch.as_tensor(batch_action,device=self.device).unsqueeze(1) #(b,1)
        batch_reward = torch.as_tensor(batch_reward,device=self.device, dtype=torch.float32) #(b)
        batch_next_state = torch.as_tensor(batch_next_state, device=self.device, dtype=torch.float32)
        batch_done = torch.as_tensor(np.float32(batch_done), device=self.device) #(b)
        
        q_values = self.policy_net(batch_state).gather(dim=1,index=batch_action) #(b,nA) -> (b,1)
        next_q_values = self.target_net(batch_next_state).max(1)[0].detach() #(b,nA) -> (b)
        expected_q_values = (batch_reward + \
                             self.gamma**self.n_multi_step * next_q_values * (1-batch_done)\
                            ).unsqueeze(1) #(b) -> (b,1)
        
        loss = torch.nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
#         for param in self.policy_net.parameters():  # clip防止梯度爆炸
#             param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.loss = loss.item()
        return self.loss
        
    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


class double_DQN(DQN):
    def __init__(self,state_dim,action_dim,cfg):
        super(double_DQN,self).__init__(state_dim,action_dim,cfg)
    def learn(self,batch_state,batch_action,batch_reward,batch_next_state,batch_done):
        if self.learning_steps % self.sync_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learning_steps += 1
        batch_state = torch.as_tensor(batch_state, device=self.device, dtype=torch.float32)
        batch_action = torch.as_tensor(batch_action,device=self.device).unsqueeze(1)
        batch_reward = torch.as_tensor(batch_reward,device=self.device, dtype=torch.float32)
        batch_next_state = torch.as_tensor(batch_next_state, device=self.device, dtype=torch.float32)
        batch_done = torch.as_tensor(np.float32(batch_done), device=self.device)
        
        # Q(s_t,a=a_t)
        q_values = self.policy_net(batch_state).gather(dim=1,index=batch_action) #(b,1)
        next_q_values = self.policy_net(batch_next_state) #(b,nA)
        next_target_values = self.target_net(batch_next_state) #(b,nA)
        # Q’(s_{t+1},a=argmax Q(s_{t+1}, a))
        next_target_q_values = next_target_values.gather(
            dim=1, index=torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1) #(b,1) ->(b)
        expected_q_values = (batch_reward + self.gamma * next_target_q_values * (1-batch_done)).unsqueeze(1) #(b,1)
        
        loss = torch.nn.MSELoss()(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
#         for param in self.policy_net.parameters():  # clip防止梯度爆炸
#             param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()


class dueling_DQN(DQN):
#     def __init__(self,state_dim,action_dim,cfg):
#         super(dueling_DQN,self).__init__(state_dim,action_dim,cfg)
    pass


class prioritized_DQN(DQN):
    def learn(self,batch_state,batch_action,batch_reward,batch_next_state,batch_done,batch_weight):
        if self.learning_steps % self.sync_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learning_steps += 1
#         batch_state,batch_action,batch_reward,batch_next_state,batch_done = self.memory.sample(self.batch_size)
        batch_state = torch.as_tensor(batch_state, device=self.device, dtype=torch.float32) #(b,c,?,?) or (b,nS)
        batch_action = torch.as_tensor(batch_action,device=self.device).unsqueeze(1) #(b,1)
        batch_reward = torch.as_tensor(batch_reward,device=self.device, dtype=torch.float32) #(b)
        batch_next_state = torch.as_tensor(batch_next_state, device=self.device, dtype=torch.float32)
        batch_done = torch.as_tensor(np.float32(batch_done), device=self.device) #(b)
        batch_weight = torch.as_tensor(batch_weight,device=self.device, dtype=torch.float32) #(b)
        
        q_values = self.policy_net(batch_state).gather(dim=1,index=batch_action).squeeze(1) #(b,nA) -> (b)
        next_q_values = self.target_net(batch_next_state).max(1)[0] #(b,nA) -> (b)
        expected_q_values = batch_reward + self.gamma * next_q_values * (1-batch_done) #(b) -> (b)
        
        prios = (q_values - expected_q_values.detach()).pow(2) * batch_weight 
        loss = prios.mean()
        self.optimizer.zero_grad()
        loss.backward()
#         for param in self.policy_net.parameters():  # clip防止梯度爆炸
#             param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return prios.detach().cpu().numpy() + 1e-8


class noisy_DQN(DQN):
    def learn(self,batch_state,batch_action,batch_reward,batch_next_state,batch_done):
        super(noisy_DQN,self).learn(batch_state,batch_action,batch_reward,batch_next_state,batch_done)
        
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        return self.loss


class ctegorical_DQN(DQN):
    def __init__(self,state_dim,action_dim,cfg):
        super(ctegorical_DQN,self).__init__(state_dim,action_dim,cfg)
        self.action_dim = action_dim
        self.Vmin = cfg.Vmin
        self.Vmax = cfg.Vmax
        self.num_atoms = cfg.num_atoms
        
    def predict(self,state):
        with torch.no_grad():
            if len(state.shape) == 4:
                state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            else:
                state = torch.as_tensor([state], device=self.device, dtype=torch.float32)
            
            dist = self.policy_net(state) * torch.linspace(self.Vmin, self.Vmax, self.num_atoms, device=self.device)
            action = dist.sum(2).max(1)[1]#.item()
        return action    

    def project_distribution(self, batch_next_state, batch_reward, batch_done):
        with torch.no_grad():
            batch_size  = batch_next_state.shape[0]

            delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1) # interval bw two atoms
            support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms, device=self.device) #(nC)

            target_prob_dist = self.target_net(batch_next_state) * support #(b,nA,nC)  p_i(x_{t+1},a) * z_i
            opt_action = target_prob_dist.sum(2).max(1)[1] #(b)  a* <- argmax Q(x_{t+1},a)
            opt_action = opt_action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.num_atoms) #(b,1,nC)
            target_prob_dist = target_prob_dist.gather(1, opt_action).squeeze(1) #(b,nC)

            batch_reward = batch_reward.unsqueeze(1).expand_as(target_prob_dist) #(b) -> (b,nC)
            batch_done   = batch_done.unsqueeze(1).expand_as(target_prob_dist) #(b) -> (b,nC)
            support = support.unsqueeze(0).expand_as(target_prob_dist) #(nC) -> (b,nC)

            Tz = batch_reward + self.gamma**self.n_multi_step * support * (1-batch_done) #(b,nC)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax) 
            b  = (Tz - self.Vmin) / delta_z #(b,nC)
            l  = b.floor()#.long()
            u  = b.ceil()#.long()

            offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=self.device)\
                            .long().unsqueeze(1).expand(batch_size, self.num_atoms)

            proj_dist = torch.zeros(target_prob_dist.shape, device=self.device) #(b,nC)
            proj_dist.reshape(-1).index_add_(0, (l.long() + offset).reshape(-1), (target_prob_dist * (u - b)).reshape(-1))
            proj_dist.reshape(-1).index_add_(0, (u.long() + offset).reshape(-1), (target_prob_dist * (b - l)).reshape(-1))
            return proj_dist
    
    def learn(self, batch_state, batch_action, batch_reward, batch_next_state, batch_done):
        if self.learning_steps % self.sync_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learning_steps += 1
        batch_state = torch.as_tensor(batch_state, device=self.device, dtype=torch.float32) #(b,c,?,?) or (b,nS)
        batch_action = torch.as_tensor(batch_action,device=self.device)\
                            .unsqueeze(1).unsqueeze(1).expand(-1, 1, self.num_atoms) #(b,1,nC)
        batch_reward = torch.as_tensor(batch_reward,device=self.device, dtype=torch.float32) #(b)
        batch_next_state = torch.as_tensor(batch_next_state, device=self.device, dtype=torch.float32)
        batch_done = torch.as_tensor(np.float32(batch_done), device=self.device) #(b)
        
        proj_dist = self.project_distribution(batch_next_state, batch_reward, batch_done) #(b,nC)
        dist = self.policy_net(batch_state) #(b,nA,nC)
        dist = dist.gather(dim=1,index=batch_action).squeeze(1) #(b,1,nC) -> (b,nC)
        dist.data.clamp_(1e-4, 1-1e-4)
        
        loss = - (proj_dist.detach() * dist.log()).sum(1).mean()
        self.optimizer.zero_grad()
        loss.backward()
#         for param in self.policy_net.parameters():  # clip防止梯度爆炸
#             param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.loss = loss.item()
        return self.loss


class rainbow_DQN(ctegorical_DQN):
    def learn(self,batch_state,batch_action,batch_reward,batch_next_state,batch_done,batch_weight):
        if self.learning_steps % self.sync_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learning_steps += 1
        batch_state = torch.as_tensor(batch_state, device=self.device, dtype=torch.float32) #(b,c,?,?) or (b,nS)
        batch_action = torch.as_tensor(batch_action,device=self.device)\
                            .unsqueeze(1).unsqueeze(1).expand(-1, 1, self.num_atoms) #(b,1,nC)
        batch_reward = torch.as_tensor(batch_reward,device=self.device, dtype=torch.float32) #(b)
        batch_next_state = torch.as_tensor(batch_next_state, device=self.device, dtype=torch.float32)
        batch_done = torch.as_tensor(np.float32(batch_done), device=self.device) #(b)
        batch_weight = torch.as_tensor(batch_weight,device=self.device, dtype=torch.float32) #(b)
        
        proj_dist = self.project_distribution(batch_next_state, batch_reward, batch_done) #(b,nC)
        dist = self.policy_net(batch_state) #(b,nA,nC)
        dist = dist.gather(dim=1,index=batch_action).squeeze(1) #(b,1,nC) -> (b,nC)
        dist.data.clamp_(1e-4, 1-1e-4)
        
        prios = - (proj_dist * dist.log()).sum(1) * batch_weight
        loss = prios.mean()
        self.optimizer.zero_grad()
        loss.backward()
#         for param in self.policy_net.parameters():  # clip防止梯度爆炸
#             param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        return prios.detach().cpu().numpy() + 1e-8

# +
# import torch.nn.functional as F
# num_actions = 6
# num_atoms = 2
# x = torch.randn([5,num_actions*num_atoms]) #(b,nA*nC)
# x = F.softmax(x.reshape(-1,num_atoms),dim=1).reshape(-1,num_actions,num_atoms) #(b,nA,nC)
# dist = x * torch.linspace(-10,10,num_atoms)
# dist.sum(2)

# +
# import os 
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# # os.environ['MASTER_ADDR'] = '172.16.10.185'              #
# # os.environ['MASTER_PORT'] = '8888'    
    
# class config:
#     def __init__(self,agent='DQN',env='CartPole-v0'):
#         self.agent = agent     # name of agent
#         self.env = env
#         self.result_path = agent+'/results/'  # path to save results
#         self.model_path = agent+'/models/'  # path to save models
#         self.train_eps = 400           # number of training episodes
#         self.eval_eps = 100            # number of evaluation episodes
#         self.memory_size = 10000       # size of replay buffer
#         self.memory_warmup_size = 200  # number of experience stored in memory before learning
#         self.learn_freq = 4            # number of steps for one learning 
#         self.sync_target_freq = 10     # number of learning steps for sync target model
#         self.batch_size = 32           # number of batches
#         self.lr = 0.01                 # learning rate
#         self.gamma = 0.99              # discount rate of reward
#         self.epsilon_start = 0.98  # start epsilon of e-greedy policy
#         self.epsilon_end = 0.01
#         self.epsilon_decay = 1/50000*16
#         self.model = 'CNN'
#         self.hidden_dim = 512 # dimmension of hidden layer
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check gpu
# cfg = config()

# +
# import matplotlib
# import matplotlib.pyplot as plt
# epsilon = lambda sample_steps: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * np.exp(-0.3 * sample_steps * cfg.epsilon_decay)
# e_greed = lambda sample_steps: max(0.01, 0.1 - 1e-6*sample_steps)
# sample_steps = list(range(300000))
# y = [epsilon(i) for i in sample_steps]
# y_ = [e_greed(i) for i in sample_steps]
# plt.plot(sample_steps,y)
# plt.plot(sample_steps,y_)
# plt.show()

# +
# device1 = torch.device('cuda:0')
# device2 = torch.device('cuda:1')
# cnn1 = Model((1,84,84),6,cfg).to(device1)
# cnn2 = Model((1,84,84),6,cfg).to(device2)

# +
# # %%time
# for _ in range(1000):
#     data = torch.randn([128*4,1,84,84]).to(device1)
#     #cnn2(data.to(device2))
#     cnn1(data)

# +
# # %%time
# for _ in range(1000):
#     data = torch.randn([128*4,1,84,84]).to(device1)
#     data = data.reshape(2,-1,1,84,84)
#     cnn1(data[0])
#     cnn2(data[1].to(device2))

# +
# import torch.nn as nn
# # nodes = 4
# # gpus = 2
# # nr = 0
# # world_size = gpus * nodes
# # rank = nr * gpus + 1

# # torch.distributed.init_process_group(backend='nccl', world_size=world_size, init_method='env://',rank=rank)
# # nn.parallel.DistributedDataParallel(mlp, device_ids=[1,2])

# #Multi-Process Single-GPU
# # torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='env://')
# # model = nn.parallel.DistributedDataParallel(cnn1, device_ids=[1], output_device=1)

# #Single-Process Multi-GPU
# # torch.distributed.init_process_group(backend="nccl")
# # model = DistributedDataParallel(model) # device_ids will include all GPU devices by default ```

# model = nn.DataParallel(Model((1,84,84),6,cfg),device_ids=[0,1],output_device=0)
# model = model.cuda()

# +
# # %%time
# for _ in range(1000):
#     data = torch.randn([128*4,1,84,84]).cuda()#.to(device1)
#     #cnn2(data.to(device2))
#     model(data)

# +


# state = np.array([5.,2.,3.,4.])
# mlp = Model(4,2,cfg).to(cfg.device)
# state = torch.tensor([state], device=cfg.device, dtype=torch.float32)
# q_values = mlp(state)
# action = q_values.max(1)[1].item()

# +
# state = np.array([1.,2.,3.,4.])
# torch.tensor([state])
# torch.from_numpy(state)

# +
# a = ReplayBuffer(2)
# a.push(np.array([1,1,1,1]),0,0,np.array([1,1,1,1]),False)
# a.push(np.array([2,2,2,2]),1,1,np.array([2,2,2,2]),True)

# +
# mlp = Model(4,2,cfg).to(cfg.device)
# batch_state,batch_action,batch_reward,batch_next_state,batch_done  = a.sample(2)
# batch_state = torch.tensor(batch_state, device=cfg.device, dtype=torch.float32)
# batch_action = torch.tensor(batch_action,device=cfg.device).unsqueeze(1)
# batch_reward = torch.tensor(batch_reward,device=cfg.device, dtype=torch.float32)
# batch_next_state = torch.tensor(batch_next_state, device=cfg.device, dtype=torch.float32)
# batch_done = torch.tensor(np.float32(batch_done), device=cfg.device)

# print(batch_state,'\n',batch_action,'\n',batch_reward,'\n',batch_next_state,'\n',batch_done,'\n')
# q_values = mlp(batch_state).gather(1,batch_action)
# print('q',q_values)
# next_q_values = mlp(batch_next_state).max(1)[0]
# print('next_q',next_q_values)
# expected_q_values = batch_reward + 1 * next_q_values * (1-batch_done)
# expected_q_values
# +
# from torchsummary import summary
# summary(mlp,(4,))
# -

