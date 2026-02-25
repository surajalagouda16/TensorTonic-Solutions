import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.array(x)   # Convert list to NumPy array
    return 1 / (1 + np.exp(-x))