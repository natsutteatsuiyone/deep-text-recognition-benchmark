import torch

class TanhExp(torch.nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))