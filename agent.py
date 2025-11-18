import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from model import LinearQNet

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, input_size=11, hidden_size=256, output_size=3,
                 lr=1e-3, gamma=0.99, max_memory=50_000, batch_size=512,
                 device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = LinearQNet(input_size, hidden_size, output_size).to(self.device)
        self.target_model = LinearQNet(input_size, hidden_size, output_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = ReplayMemory(max_memory)
        self.batch_size = batch_size
        self.gamma = gamma

        self.n_games = 0
        self.epsilon = 0  # will be set dynamically
        self.output_size = output_size

    def get_action(self, state):
        # epsilon-greedy
        self.epsilon = max(0, 80 - self.n_games)  # simple annealing
        if random.random() < self.epsilon / 200:
            move = random.randint(0, self.output_size - 1)
            action = [0] * self.output_size
            action[move] = 1
            return action

        state0 = torch.tensor(state, dtype=torch.float32).to(self.device)
        if state0.dim() == 1:
            state0 = state0.unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        action = [0] * self.output_size
        action[move] = 1
        return action

    def remember(self, state, action, reward, next_state, done):
        act_idx = int(np.argmax(action))
        self.memory.push((state, act_idx, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.uint8).unsqueeze(1).to(self.device)

        q_values = self.model(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target = rewards + (1 - dones.float()) * self.gamma * next_q

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path, extra=None):
        cfg = {
            'input_size': self.model.linear1.in_features,
            'hidden_size': self.model.linear1.out_features,
            'output_size': self.model.linear3.out_features
        }
        torch.save({
            'cfg': cfg,
            'state_dict': self.model.state_dict(),
            'extra': extra
        }, path)

    def load(self, path):
        cp = torch.load(path, map_location=self.device)
        cfg = cp['cfg']
        self.model = LinearQNet(cfg['input_size'], cfg['hidden_size'], cfg['output_size']).to(self.device)
        self.model.load_state_dict(cp['state_dict'])
        self.target_model = LinearQNet(cfg['input_size'], cfg['hidden_size'], cfg['output_size']).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
