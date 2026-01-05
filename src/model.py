import os
import tensorflow as tf


def load_trained_model(path):
    """
    Load a trained Keras model from disk.
    
    Args:
        path: Path to the model file (.keras or .h5)
        
    Returns:
        Loaded Keras model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if not (path.endswith(".keras") or path.endswith(".h5")):
        raise ValueError(
            "Keras 3 only supports V3 `.keras` files or legacy `.h5` files. "
            f"Given: {path}"
        )

    model = tf.keras.models.load_model(path)
    print("Loaded model from", path)
    return model
