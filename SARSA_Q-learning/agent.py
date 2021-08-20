# -*- coding: utf-8 -*-
import numpy as np


class QLearning(object):
    def __init__(self,state_dim,action_dim,cfg):
        self.state_dim = state_dim    # dimension of state
        self.action_dim = action_dim  # dimension of action
        self.lr = cfg.lr              # learning rate
        self.gamma = cfg.gamma        # discount rate of reward
        self.sample_count = 0  
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q = np.zeros((state_dim, action_dim)) 
    
    def sample(self,state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.sample_count / self.epsilon_decay)
        if np.random.uniform(0, 1) > self.epsilon:  # 随机选取0-1之间的值，如果大于epsilon就按照贪心策略选取action，否则随机选取
            action = self.predict(state)
        else:
            action = np.random.choice(self.action_dim)  #有一定概率随机探索选取一个动作
        return action
    
    def predict(self, state):
        '''根据输入观测值，采样输出的动作值，带探索，测试模型时使用
        '''
        Q_list = self.Q[state, :]
        Q_max = np.max(Q_list)
        action_list = np.where(Q_list == Q_max)[0]  
        action = np.random.choice(action_list) # Q_max可能对应多个 action ，可以随机抽取一个
        return action
    
    def learn(self, state, action, reward, next_state, done):
        current_Q = self.Q[state, action]
        if done:
            target_Q = reward  # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_state, :])  # Q-learning
        self.Q[state, action] += self.lr * (target_Q - current_Q)  # update q
        
    def save(self,path):
        np.save(path+"Q_table.npy", self.Q)
    def load(self, path):
        self.Q = np.load(path+"Q_table.npy")


class SARSA(QLearning):
    def learn(self, state, action, reward, next_state, next_action, done):
        current_Q = self.Q[state, action]
        if done:
            target_Q = reward  # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * self.Q[next_state, next_action]  # SARSA
        self.Q[state, action] += self.lr * (target_Q - current_Q)  # update q


