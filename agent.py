import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import backtrader as bt

# Define the Q-Network
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

# Experience Replay Memory
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

# DQN Agent
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
        action_batch = torch.tensor(action_batch, dtype=torch.int64)  # Convert to tensor
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)  # Convert to tensor
        next_state_batch = torch.stack(next_state_batch)
        
        q_values = self.q_network(state_batch)
        print("Shape of q_values:", q_values.shape)  # Print the shape of q_values

        # Squeeze the unnecessary middle dimension from q_values if it exists
        if q_values.dim() == 3 and q_values.size(1) == 1:
            q_values = q_values.squeeze(1)
        print("New shape of q_values after squeeze:", q_values.shape)

        actions = action_batch.unsqueeze(1)
        print("Shape of actions after unsqueeze:", actions.shape)  # Print the shape of actions after unsqueeze

        q_values = q_values.gather(1, actions).squeeze(1)
        print("Shape of q_values after gather and squeeze:", q_values.shape)  # Print the shape after gather and squeeze

        next_q_values = self.target_network(next_state_batch)
        print("Shape of next_q_values before max operation:", next_q_values.shape)

        # Ensure we are taking the max across the correct dimension
        next_q_values = next_q_values.max(2)[0].detach()  # max along the action dimension
        print("Shape of next_q_values after max operation:", next_q_values.shape)

        print("Shape of reward_batch:", reward_batch.shape)
        print("Shape of next_q_values:", next_q_values.shape)

        target_q_values = reward_batch + (self.gamma * next_q_values)

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
