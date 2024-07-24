import jax.numpy as jnp
from jax import value_and_grad

class Loss:
    def __init__(self, value, prev, grad_fn):
        self.value = value
        self.prev = prev
        self.grad_fn = grad_fn
    
    def backward(self, lr):
        for pair in zip(self.prev, self.grad_fn(self.value)):
            pair[0].backward(pair[1], lr)

def cross_entropy(y_pred_tensor, y_true):
    def calculate_cross_entropy_loss(y_pred, y_true):
        y_pred = jnp.exp(y_pred - jnp.max(y_pred))
        y_pred = y_pred / jnp.sum(y_pred, axis=-1, keepdims=True)
        loss = -jnp.sum(y_true * jnp.log(y_pred)) / len(y_pred)
        return loss
    
    loss, grad_fn = value_and_grad(calculate_cross_entropy_loss)(y_pred_tensor.tensor, y_true)
    
    return Loss(loss, [y_pred_tensor], grad_fn)
    
    