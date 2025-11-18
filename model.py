import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint['cfg']
    model = LinearQNet(cfg['input_size'], cfg['hidden_size'], cfg['output_size']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model
