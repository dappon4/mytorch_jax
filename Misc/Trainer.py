from sklearn.datasets import fetch_openml
import jax.numpy as jnp

def load_mnist(flatten=True):
    print("loading MNIST...")
    digits = fetch_openml('mnist_784')
    
    data = jnp.array(digits.data)
    data = data / 255
    
    target = jnp.eye(10)[digits.target]

    if not flatten:
        data = data.reshape(-1, 1, 28, 28)
    else:
        data = data.reshape(-1, 1, 784)
    
    print("MNIST loaded")
    return data, target