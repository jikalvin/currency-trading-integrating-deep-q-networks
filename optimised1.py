import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import backtrader as bt
from datetime import datetime
import yfinance as yf


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, state, action, reward, next_state):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters())
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                return self.q_network(state).argmax().item()

    def remember(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch = zip(*batch)
        
        state_batch = torch.stack(state_batch)
        action_batch = torch.tensor(action_batch).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch)
        next_state_batch = torch.stack(next_state_batch)
        
        q_values = self.q_network(state_batch)
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        
        # Gather the Q-values corresponding to the taken actions
        q_values = q_values.gather(1, action_batch).squeeze(1)
        
        target_q_values = reward_batch + (self.gamma * next_q_values)
        
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

class DQNStrategy(bt.Strategy):
    params = dict(
        state_dim=4,  # Updated state dimension to match get_state function
        action_dim=3
    )
    
    def __init__(self):
        self.agent = DQNAgent(self.params.state_dim, self.params.action_dim)
        self.dataclose = self.datas[0].close
        self.order = None
    
    def next(self):
        state = self.get_state()
        action = self.agent.act(state)
        reward = 0
        
        if action == 1 and self.order is None:
            self.order = self.buy()
        elif action == 2 and self.order is None:
            self.order = self.sell()
        elif action == 0 and self.order:
            self.close()
            reward = self.dataclose[0] - self.order.executed.price if self.order.isbuy() else self.order.executed.price - self.dataclose[0]
            self.order = None

        next_state = self.get_state()
        self.agent.remember(state, action, reward, next_state)
        self.agent.replay()
        self.agent.update_target_network()

    def get_state(self):
        state = np.array([self.dataclose[0], self.datas[0].high[0], self.datas[0].low[0], self.datas[0].open[0]])
        return torch.tensor(state, dtype=torch.float).unsqueeze(0)

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(DQNStrategy)
    
    data = bt.feeds.PandasData(dataname=yf.download('AAPL', start='2010-01-01', end='2020-12-31', auto_adjust=True))
    cerebro.adddata(data)
    
    cerebro.broker.set_cash(100000)
    cerebro.run()
    cerebro.plot()