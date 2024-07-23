import jax
from jax import value_and_grad, grad, jit, vmap
import jax.numpy as jnp
import random

class Tensor:
    def __init__(self, tensor, prev=None, grad_fns=None) -> None:
        self.tensor = tensor
        self.grad_fns = grad_fns
        self.shape = tensor.shape
        
        if not prev:
            self.prev = []
        else:
            self.prev = prev
    
    def __repr__(self) -> str:
        return f"Tensor({self.tensor})"
    
    def __add__(self, other):
        return Tensor(self.tensor + other.tensor, [self, other], lambda x: (x,x))
    
    def __mul__(self, num):
        if type(num) != int:
            raise RuntimeError("Only multiplication by scalar is supported")
        
        return Tensor(self.tensor * num, [self], lambda x: num*x)
    
    def __rmul__(self, num):
        return self.__mul__(num)
    
    def backward(self, grad):
        if self.grad_fn:
            for pair in zip(self.prev, self.grad_fn(grad)):
                pair[0].backward(pair[1])
    
    @staticmethod            
    def zeros(*shape):
        return Tensor(jnp.zeros(shape))
    
    @staticmethod
    def ones(*shape):
        return Tensor(jnp.ones(shape))
    
    @staticmethod
    def random_init(*shape):
        key = jax.random.PRNGKey(random.randint(0,100))
        return Tensor(jax.random.uniform(key, shape))
    
    @staticmethod
    def xavier_init(*shape):
        assert len(shape) == 2
        key = jax.random.PRNGKey(random.randint(0,100))
        std = jnp.sqrt(2/shape[0]+shape[1])
        return Tensor(std * jax.random.normal(key, shape))