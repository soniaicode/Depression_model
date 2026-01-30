"""
Memory optimization utilities for Render deployment
"""
import gc
import os

def optimize_memory():
    """Force garbage collection to free memory"""
    gc.collect()
    
def get_memory_usage():
    """Get current memory usage in MB"""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_model_lazy(model_path):
    """Load model only when needed and unload after use"""
    import joblib
    model = joblib.load(model_path)
    return model

def clear_tensorflow_session():
    """Clear TensorFlow session to free GPU/CPU memory"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        keras.backend.clear_session()
        gc.collect()
    except:
        pass
