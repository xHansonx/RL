# -*- coding: utf-8 -*-
import torch.nn as nn


# +
class Model(nn.Module):
    ''' 多层感知机
        输入：state维度
        输出：概率
    '''
    def __init__(self,state_dim,hidden_dim = 36):
        super(MLP, self).__init__()
        # 24和36为hidden layer的层数，可根据state_dim, action_dim的情况来改变
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Prob of Left

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        return x

# class Model(object):
#     def __init__():
#         pass
#     def dayin():
#         print(1)

# +
# if __name__ == '__main__':
#     model()
# -

