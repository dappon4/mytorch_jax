import mytorch_jax as torch
import mytorch_jax.nn as nn
from mytorch_jax.Tensor import Tensor

class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.W = Tensor.xavier_init(input_size, output_size)
        self.b = None
    
    def forward(self, x):
        x = torch.matmul(x, self.W)
        if self.b is None:
            self.b = Tensor.ones(x.shape)
        
        return x + self.b
        