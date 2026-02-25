import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Positions (seq_len, 1)
    positions = np.arange(seq_len)[:, np.newaxis]

    # Dimensions (1, d_model)
    dims = np.arange(d_model)[np.newaxis, :]

    # Compute angle rates
    angle_rates = 1 / (base ** (2 * (dims // 2) / d_model))

    # Compute angles
    angles = positions * angle_rates

    # Initialize output
    PE = np.zeros((seq_len, d_model))

    # Apply sin to even indices
    PE[:, 0::2] = np.sin(angles[:, 0::2])

    # Apply cos to odd indices
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    return PE