# import random
import numpy as np
# import collections

# +
# class ReplayBuffer(object):
#     def __init__(self,memory_size):
#         self.buffer = collections.deque(maxlen=memory_size)
#     def push(self,state,action,reward,next_state,done):
#         self.buffer.append((state,action,reward,next_state,done))
#     def sample(self,batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         batch_state,batch_action,batch_reward,batch_next_state,batch_done = zip(*batch)
#         return batch_state,batch_action,batch_reward,batch_next_state,batch_done
#     def __len__(self):
#         return len(self.buffer)
# -

class ReplayBuffer(object):
    
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.idx = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.idx] = (state, action, reward, next_state, done)
        self.idx = (self.idx + 1) % self.size
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done =  zip(*samples)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.buffer     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha # prios cannot be negative which would result to complex number of p
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False) # p: non-negative
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        states = np.concatenate(states)
        next_states = np.concatenate(next_states)
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = max(1e-8,prio)

    def __len__(self):
        return len(self.buffer)


class NStepReplayBuffer(ReplayBuffer):
    def __init__(self, nstep, discount_factor, buffer):
        assert nstep >= 1
        assert discount_factor >= 0
        self.nstep = nstep
        self.discount_factor = discount_factor
        self.buffer = buffer
        
        self._states = []
        self._actions = []
        self._rewards = []
        self._nstep_reward = 0.
        
    def push(self, state, action, reward, next_state, done):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._nstep_reward += (self.discount_factor ** (len(self._states) - 1)) * reward

        if done:
            while self._states:
                self.push2memory(self._states[0], self._actions[0], self._nstep_reward, next_state, done)
            self._nstep_reward = 0.
            
        if len(self._states) == self.nstep:
            self.push2memory(self._states[0], self._actions[0], self._nstep_reward, next_state, done)

    def push2memory(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self._nstep_reward = self._nstep_reward - self._rewards[0]
        self._nstep_reward /= self.discount_factor 
        del self._states[0]
        del self._actions[0]
        del self._rewards[0]

    def sample(self, *args, **kwargs):
        return self.buffer.sample(*args, **kwargs)

    def update_priorities(self, *args, **kwargs):
        return self.buffer.update_priorities(*args, **kwargs)

    def __len__(self):
        return len(self.buffer)
