import jax.numpy as jnp
from jax import grad, jit, vmap

class Module:
    def __init__(self) -> None:
        self.training = True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    def forward(self, x):
        return x
    
    def train(self) -> None:
        self.training = True
    
    def eval(self) -> None:
        self.training = False
    
    