import torch
import torch.nn as nn


class ActNorm(nn.Module):
    EPSILON = 1e-6

    def __init__(self, shape: tuple):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(shape))
        self.sigma = nn.Parameter(torch.ones(shape))
        self.is_initialized = False

    def forward(self, x, reverse=False):
        if not self.is_initialized:
            self.mu.data = x.transpose(0, 1).view(x.size(1), -1).mean(1).view_as(self.mu)
            self.sigma.data = (x.transpose(0, 1).view(x.size(1), -1).std(1) + self.EPSILON).view_as(self.sigma)
            self.is_initialized = True

        if not reverse:
            out = (x - self.mu) / self.sigma
            log_det = -self.sigma.abs().log().sum() * x.size(2) * x.size(3)
        else:
            out = x * self.sigma + self.mu
            log_det = self.sigma.abs().log().sum() * x.size(2) * x.size(3)

        return out, log_det
