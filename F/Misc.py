from jax import value_and_grad, grad, jit, vmap
from mytorch_jax.Tensor import Tensor

def matmul(tensor1, tensor2):
    def matmul(tensor1, tensor2):
        new_tensor, *grad_fn = value_and_grad(jnp.matmul, (0,1))(tensor1.tensor, tensor2.tensor)
        
        return Tensor(new_tensor, [tensor1, tensor2], grad_fn)