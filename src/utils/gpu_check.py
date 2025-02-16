import tensorflow as tf
import os

def setup_gpu():
    """Configure GPU settings and verify setup"""
    # Set TF log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Enable mixed precision for better performance on RTX cards
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Configure memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Set memory limit to 90% of available memory
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=14738)]  # 14.7GB for RTX 4070
                )
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("\nNo GPU found. Please check:")
        print("1. NVIDIA driver version 551.23 is installed")
        print("2. CUDA 12.4 is installed")
        print("3. cuDNN compatible with CUDA 12.4 is installed")
        print("4. Environment variables are set correctly:")
        print("   CUDA_PATH = C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4")
        print("   Add to PATH:")
        print("   - %CUDA_PATH%\\bin")
        print("   - %CUDA_PATH%\\libnvvp\n")

if __name__ == "__main__":
    setup_gpu()
    
    # Test GPU
    print("\nTesting GPU computation:")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
            c = tf.matmul(a, b)
            print("GPU test successful!")
            print(f"Test computation result:\n{c.numpy()}")
    except:
        print("GPU test failed!")
