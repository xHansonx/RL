# -*- coding: utf-8 -*-
import numpy as np
import torch 
from torch.distributions import Categorical, Normal
from torch.nn.utils import clip_grad_norm_
from model import Model
# from memory import ReplayBuffer

def Agent(state_dim,action_dim,cfg):
    if cfg.agent == 'A2C':
        return A2C(state_dim,action_dim,cfg)
    elif cfg.agent == 'GAE':
        return GAE(state_dim,action_dim,cfg)
    elif cfg.agent == 'PPO':
        return PPO(state_dim,action_dim,cfg)
    elif cfg.agent == 'DDPG':
        return DDPG(state_dim,action_dim,cfg)


class A2C(object):
    def __init__(self,state_dim,action_dim,cfg):
        self.agent_name = cfg.agent
        self.n_multi_step = cfg.n_multi_step          # number of steps of multi-step style gain
        self.gamma = cfg.gamma                        # discount rate of reward
        self.lr = cfg.lr                              # learning rate
        self.state_dim = state_dim                    # dimension of state
        self.action_dim = action_dim                  # dimension of action
        self.w_critic_loss = cfg.w_critic_loss                      # weight of critic loss
        self.w_entropy_loss = cfg.w_entropy_loss                   # weight of entropy loss
#         self.sample_steps = 0                         # number of steps of sample()
        self.learning_steps = 0                       # number of steps of learning
        self.device = cfg.device                      # cpu or gpu
        self.a2c_net = Model(state_dim,action_dim,cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.a2c_net.parameters(), lr=cfg.lr)
        self.model = cfg.model
        
    def sample(self,state):
        value,dist = self.predict(state)
        action = dist.sample()
        return value, dist, action
    
    def predict(self,state):
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        value,dist = self.a2c_net(state)
        return value,dist  
     
    def learn(self,buffer,next_states):
        self.learning_steps += 1
        
        values, dists, actions, rewards, dones = zip(*buffer)
        next_value,_ = self.predict(next_states)  #(nEnv,1)
        R = next_value #(nEnv,1)
        gains = []
        for r,done in zip(rewards[::-1],dones[::-1]):
            R = r + self.gamma * R * (1-done) 
            gains.append(R)
        gains = gains[::-1] # [(nE,1)]*nT
        
        log_probs = [dist.log_prob(action) for dist,action in zip(dists,actions)] # [(nE,1)]*nT or [(nE,Dim_A)]*nT
        entropys = [dist.entropy() for dist in dists] # [(nE,1)]*nT or [(nE,Dim_A)]*nT
        
        batch_gains,batch_values,batch_log_probs,batch_entropys = map(
            lambda x:torch.cat(x),[gains,values,log_probs,entropys])
        # Q(s_t,a_t) - V(s_t) = r_t + gamma*V(s_{t+1}) - V(s_t)
        advantage = batch_gains.detach() - batch_values #(nEnv*nStep,1)
        # ∇R = ∑_nstep ∑_t (Q(s_t,a_t) - V(s_t)) * ∇log p(a_t|s_t)
        if 'Discrete' in self.model:
            loss_actor = -(batch_log_probs.unsqueeze(1) * advantage.detach()).mean()
        else:
            loss_actor = -(batch_log_probs * advantage.detach()).mean()
        loss_critic = advantage.pow(2).mean()
        # max entropy -> min -entropy
        loss = loss_actor + self.w_critic_loss*loss_critic - self.w_entropy_loss*batch_entropys.mean()
        self.optimizer.zero_grad()
        loss.backward()
#         for param in self.a2c_net.parameters():  # clip防止梯度爆炸
#             param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.loss = loss.item()
        return self.loss
        
    def save(self, path):
        torch.save(self.a2c_net.state_dict(), path+self.agent_name+'_checkpoint.pth')

    def load(self, path):
        self.a2c_net.load_state_dict(torch.load(path+self.agent_name+'_checkpoint.pth'))


class GAE(A2C):
    def __init__(self,state_dim,action_dim,cfg):
        super(GAE,self).__init__(state_dim,action_dim,cfg)
        self.tau = cfg.tau
        
    def learn(self,buffer,next_states):
        self.learning_steps += 1
        
        values, dists, actions, rewards, dones = zip(*buffer) # 
        next_value,_ = self.predict(next_states)  #(nEnv,1)
        values += (next_value,)
        R = 0
        gains = []
        for i in range(len(rewards)-1,-1,-1):
            delta = rewards[i] + self.gamma * values[i+1]*(1-dones[i]) - values[i]
            R = delta + self.gamma * self.tau * R * (1-dones[i]) 
            gains.append(R)
        gains = gains[::-1] # [(nE,1)]*nT
        
        log_probs = [dist.log_prob(action) for dist,action in zip(dists,actions)] # [(nE,1)]*nT or [(nE,Dim_A)]*nT 
        entropys = [dist.entropy() for dist in dists] # [(nE,1)]*nT or [(nE,Dim_A)]*nT
        
        batch_gains,batch_values,batch_log_probs,batch_entropys = map(
            lambda x:torch.cat(x),[gains,values[:-1],log_probs,entropys])
        # Q(s_t,a_t) - V(s_t) = r_t + gamma*V(s_{t+1}) - V(s_t)
        advantage = batch_gains.detach() - batch_values # (nEnv*nStep,1)
        # ∇R = ∑_nstep ∑_t (Q(s_t,a_t) - V(s_t)) * ∇log p(a_t|s_t)
        if 'Discrete' in self.model:
            loss_actor = -(batch_log_probs.unsqueeze(1) * advantage.detach()).mean()
        else:
            loss_actor = -(batch_log_probs * advantage.detach()).mean()
#         loss_actor = -(batch_log_probs * advantage.detach()).mean()
        loss_critic = advantage.pow(2).mean()
        # max entropy -> min -entropy
        loss = loss_actor + self.w_critic_loss*loss_critic - self.w_entropy_loss*batch_entropys.mean()
        self.optimizer.zero_grad()
        loss.backward()
#         for param in self.a2c_net.parameters():  # clip防止梯度爆炸
#             param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.loss = loss.item()
        return self.loss


class PPO(A2C):
    def __init__(self,state_dim,action_dim,cfg):
        super(PPO,self).__init__(state_dim,action_dim,cfg)
        self.n_ppo_epochs = cfg.n_ppo_epochs
        self.mini_batch_size = cfg.mini_batch_size      
        self.ratio_clip_epsilon = cfg.ratio_clip_epsilon  
        
    def learn(self,buffer,next_states):
        self.learning_steps += 1
        
        states, values, dists, actions, rewards, dones = zip(*buffer)
        next_value,_ = self.predict(next_states) #(nEnv,1)
        R = next_value #(nEnv,1)
        gains = []
        for r,done in zip(rewards[::-1],dones[::-1]):
            R = r + self.gamma * R * (1-done) 
            gains.append(R)
        gains = gains[::-1] # [(nE,1)]*nT
        log_probs = [dist.log_prob(action) for dist,action in zip(dists,actions)] # [(nE,1)]*nT or [(nE,Dim_A)]*nT
        batch_states,batch_actions,batch_gains,batch_values,batch_log_probs = map(
            lambda x:torch.cat(x),[states,actions,gains,values,log_probs])
        # Aθ = Q(s_t,a_t) - V(s_t) = r_t + gamma*V(s_{t+1}) - V(s_t)
        batch_advantages = batch_gains.detach() - batch_values.detach() #(nEnv*nStep,1)

        batch_size = batch_advantages.shape[0]
        losses = []
        for _ in range(self.n_ppo_epochs):
            for _ in range(batch_size//self.mini_batch_size):
                idx = np.random.randint(0, batch_size, self.mini_batch_size)
                _batch_states, _batch_actions, _batch_log_probs, _batch_gains, _batch_advantages =\
                    batch_states[idx], batch_actions[idx], batch_log_probs[idx], batch_gains[idx], batch_advantages[idx]
                __batch_values , __batch_dists = self.predict(_batch_states)
                __batch_entropys = __batch_dists.entropy().mean()
                __batch_log_probs = __batch_dists.log_prob(_batch_actions)

                # ∇f(x) = f(x)∇logf(x)
                # ∇Rθ' = ∑_nstep ∑_t  pθ'(a_t|s_t)/pθ(a_t|s_t) * Aθ * ∇log pθ'(a_t|s_t)
                # ∇Rθ' = ∑_nstep ∑_t  ∇pθ'(a_t|s_t)/pθ(a_t|s_t) * Aθ 
                # ∇Rθ' = ∑_nstep ∑_t min( ∇pθ'(a_t|s_t)/pθ(a_t|s_t)*Aθ, clip[∇pθ'(a_t|s_t)/pθ(a_t|s_t),1-ε,1+ε]*Aθ )     
                ratio = (__batch_log_probs - _batch_log_probs.detach()).exp()
                ratio = ratio.unsqueeze(1) if 'Discrete' in self.model else ratio
                j1 = ratio * _batch_advantages.detach()
                j2 = torch.clamp(ratio, 1.0-self.ratio_clip_epsilon, 1+ self.ratio_clip_epsilon) * _batch_advantages.detach()

                loss_actor = - torch.min(j1,j2).mean()
                loss_critic = (_batch_gains.detach() - __batch_values).pow(2).mean()
                # max entropy -> min -entropy
                loss = loss_actor + self.w_critic_loss*loss_critic - self.w_entropy_loss*__batch_entropys

                self.optimizer.zero_grad()
                loss.backward()
                #         for param in agent.a2c_net.parameters():  # clip防止梯度爆炸
                #             param.grad.data.clamp_(-1, 1)
                self.optimizer.step()  
                losses.append(loss.item())
        self.loss = np.mean(losses)
        return self.loss


# +
from model import Actor_MLP, Critic_MLP

class DDPG(object):
    def __init__(self,state_dim,action_dim,cfg):
        self.agent_name = cfg.agent
#         self.n_multi_step = cfg.n_multi_step          # number of steps of multi-step style gain
        self.gamma = cfg.gamma                        # discount rate of reward
        self.actor_lr = cfg.actor_lr                  # learning rate of policy opt
        self.critic_lr = cfg.critic_lr                # learning rate of value opt
        self.state_dim = state_dim                    # dimension of state
        self.action_dim = action_dim                  # dimension of action     
        self.sample_steps = 0                         # number of steps of sample()
        self.learning_steps = 0                       # number of steps of learning
        self.device = cfg.device                      # cpu or gpu
        self.epsilon =  lambda sample_steps: cfg.epsilon_end + \
                                            (cfg.epsilon_start - cfg.epsilon_end) * \
                                            np.exp(-0.1 * sample_steps * cfg.epsilon_decay)
        self.max_act = cfg.max_act                    # max action value
        self.soft_tau = cfg.soft_tau                  # soft update target model
        
        self.policy_net = Actor_MLP(state_dim,action_dim,cfg.hidden_dim,cfg.max_act).to(self.device)
        self.value_net = Critic_MLP(state_dim,action_dim,cfg.hidden_dim).to(self.device)
        self.target_policy_net = Actor_MLP(state_dim,action_dim,cfg.hidden_dim,cfg.max_act).to(self.device)
        self.target_value_net = Critic_MLP(state_dim,action_dim,cfg.hidden_dim).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.actor_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=cfg.critic_lr)
        ## copy params to target net ##
        self.sync_target_net(self.target_policy_net, self.policy_net, tau=1)
        self.sync_target_net(self.target_value_net, self.value_net, tau=1)
            
    def sample(self,state):
        with torch.no_grad():
            action = self.predict(state)
            sigma = self.epsilon(self.sample_steps)
            noise = Normal(torch.zeros_like(action),torch.ones_like(action)*sigma)
            action = torch.clamp(action+noise.sample(),-self.max_act,self.max_act) 
            self.sample_steps += 1
        return action.cpu().numpy()
    
    def predict(self,state):
        with torch.no_grad():
            state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            action = self.policy_net(state)
        return action    
    
    def learn(self,batch_state,batch_action,batch_reward,batch_next_state,batch_done):
        self.learning_steps += 1
        
        batch_state = torch.as_tensor(batch_state, device=self.device, dtype=torch.float32) #(b,c,?,?) or (b,nS)
        batch_action = torch.as_tensor(batch_action,device=self.device) #(b,nA)
        batch_reward = torch.as_tensor(batch_reward,device=self.device, dtype=torch.float32).unsqueeze(1) #(b,1)
        batch_next_state = torch.as_tensor(batch_next_state, device=self.device, dtype=torch.float32)
        batch_done = torch.as_tensor(np.float32(batch_done), device=self.device).unsqueeze(1) #(b,1)
        
        value = self.value_net(batch_state,batch_action) #(b,1)
        next_act = self.target_policy_net(batch_next_state) #(b,nA)
        next_value = self.target_value_net(batch_next_state,next_act) #(b,1)
#         expected_value = batch_reward + self.gamma**self.n_multi_step*next_value*(1-batch_done)
        expected_value = batch_reward + self.gamma*next_value*(1-batch_done)
        
        loss_actor = -self.value_net(batch_state,self.policy_net(batch_state)).mean() #(b,1) -> (1)
        loss_critic = torch.nn.MSELoss()(value, expected_value.detach())
        loss = loss_actor + .5*loss_critic 
        
        self.policy_optimizer.zero_grad()
        loss_actor.backward()
#         for param in self.policy_net.parameters():  # clip防止梯度爆炸
#             param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()
        self.value_optimizer.zero_grad()
        loss_critic.backward()
#         for param in self.value_net.parameters():  # clip防止梯度爆炸
#             param.grad.data.clamp_(-1, 1)
        self.value_optimizer.step()
        
        self.sync_target_net(self.target_policy_net, self.policy_net, tau=self.soft_tau)
        self.sync_target_net(self.target_value_net, self.value_net, tau=self.soft_tau)
        self.loss = loss.item()
        return self.loss
    
    def sync_target_net(self, target, source, tau=0):
        for target_param, param in zip(target.parameters(), source.parameters()):
#             target_param.detach_()
            target_param.data.copy_(target_param * (1.0 - tau) + param * tau)
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path+self.agent_name+'_checkpoint.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path+self.agent_name+'_checkpoint.pth'))
#         for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
#             param.data.copy_(target_param.data)

# +
# class config:
#     def __init__(self):
#         self.seed = 1                         # seed
#         self.agent = 'DDPG'             # name of agent
#         self.env_id = 'Pendulum-v0'    # name of environment
#         self.n_train_env = 16           # number of training environment 
#         self.n_eval_env = 4           # number of evaluation environment 
#         self.max_train_frames = 1e6     # max number of training frames
#         self.max_eval_frames = 1e4      # max number of evaluation frames
#         self.memory_size = 100000       # size of replay buffer
#         self.memory_warmup_size = 600  # number of experience stored in memory before learning
#         self.learn_freq = 2            # number of steps for one learning 
#         self.sync_target_freq = 32     # number of learning steps for sync target model
#         self.batch_size = 512          # number of examples in one batch
#         self.gamma = 0.99              # discount rate of reward
#         self.n_multi_step = 2          # number of steps of multi-step style gain 
#         self.soft_tau = 0.01           # parm for soft updating target model
#         self.actor_lr = 0.0001         # learning rate of policy opt
#         self.critic_lr = 0.0001        # learning rate of value opt
#         self.gamma = 0.90              # discount rate of reward
#         self.n_multi_step = 1          # number of steps of multi-step style gain 
#         self.epsilon_start = 0.90      # starting epsilon of e-greedy policy
#         self.epsilon_end = 0.001        # ending epsilon
#         self.epsilon_decay = 1/5000    # e = e_end + (e_start - e_end) * exp(-1. * sample_steps * e_decay)
#         self.model = 'Continuous'       # name or type of model
#         self.hidden_dim = 256           # dimmension of hidden layer
#         self.max_act = 2.0
#         self.device = torch.device("cpu")  # check gpu

# +
# cfg = config()
# DDPG(8,2,cfg)

# +
# import torch.nn as nn
# m = nn.Linear(hidden_dim, act_dim)
# +
# policy_net = Actor_MLP(8,2)
# value_net = Critic_MLP(8,2)
# s = torch.rand(5,8)
# a = policy_net(s)
# value_net(s,policy_net(s)).mean()
# -



# +
# epsilon_exp =  lambda sample_steps: 0.001 + (0.9 - 0.001) * np.exp(-0.1 * sample_steps * 1/5000)
# epsilon_linear =  lambda sample_steps: 0.9 - (0.9 - 0.001) * min(1.0, sample_steps/100000)
# import matplotlib.pyplot as plt
# x = np.arange(1,150000)
# y = list(map(epsilon_exp,x))
# y_ = list(map(epsilon_linear,x))
# plt.plot(x,y)
# plt.plot(x,y_)
# -


