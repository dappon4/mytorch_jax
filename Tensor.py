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
    
    @staticmethod
    def matmul(tensor1, tensor2):
        new_tensor, *grad_fn = value_and_grad(jnp.matmul, (0,1))(tensor1.tensor, tensor2.tensor)
        
        return Tensor(new_tensor, [tensor1, tensor2], grad_fn)