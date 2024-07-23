from jax import value_and_grad, grad, jit, vmap
import jax.numpy as jnp

class Tensor:
    def __init__(self, tensor, prev=None, grad_fn=None) -> None:
        self.tensor = tensor
        self.grad_fn = grad_fn
        
        if not prev:
            self.prev = []
        else:
            self.prev = prev
    
    def __repr__(self) -> str:
        return f"Tensor({self.tensor})"
    
    def __add__(self, other):
        return Tensor(self.tensor + other.tensor, [self, other])
    
    def __mul__(self, other):
        if type(other) != int:
            raise RuntimeError("Only multiplication by scalar is supported")
        
        return Tensor(self.tensor * other, [self])
    
    def __rmul__(self, other):
        return self.__mul__(other)