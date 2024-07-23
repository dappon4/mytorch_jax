from mytorch_jax.Tensor import Tensor
from jax import value_and_grad

def relu(x: Tensor) -> Tensor:
    value, grad_fn = value_and_grad((lambda x: x if x > 0 else 0))(x.tensor)
    
    return Tensor(value, [x], grad_fn)