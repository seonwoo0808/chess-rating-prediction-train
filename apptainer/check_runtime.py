"""CPU-safe build check, with an explicit GPU execution check for compute nodes."""
import argparse
import ctypes
import importlib.metadata as metadata
import json
import sys

import numba
import pyarrow
import tensorflow as tf
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    assert sys.version_info[:3] == (3, 12, 12), sys.version
    assert tf.__version__ == '2.20.0', tf.__version__
    assert tf.test.is_built_with_cuda(), 'TensorFlow wheel lacks CUDA support'
    assert int(keras.__version__.split('.')[0]) >= 3, keras.__version__
    assert not keras.__name__.startswith('tf_keras'), keras.__name__
    try:
        metadata.version('tf-keras')
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError('tf-keras must not be installed in this image')
    nvidia_wheels = [d.metadata['Name'] for d in metadata.distributions()
                     if d.metadata['Name'].lower().startswith('nvidia-')]
    assert not nvidia_wheels, f'Duplicate NVIDIA Python packages: {nvidia_wheels}'
    # Loading and querying cuDNN does not require an allocated GPU.
    cudnn = ctypes.CDLL('libcudnn.so.9')
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    cudnn_version = cudnn.cudnnGetVersion()
    assert 90300 <= cudnn_version < 100000, cudnn_version
    print(json.dumps({
        'tensorflow': tf.__version__, 'keras': keras.__version__,
        'keras_module': keras.__name__, 'numba': numba.__version__,
        'pyarrow': pyarrow.__version__, 'build': tf.sysconfig.get_build_info(),
        'cudnn_runtime': cudnn_version,
        'gpus': [d.name for d in tf.config.list_physical_devices('GPU')],
    }, indent=2))
    if args.gpu:
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('No GPU detected; check allocation, --nv and host driver')
        tf.config.set_soft_device_placement(False)
        with tf.device('/GPU:0'):
            # Exercise cuBLAS and cuDNN, including backward execution.
            x = tf.ones((2, 8, 8, 12))
            kernel = tf.Variable(tf.ones((3, 3, 12, 32)))
            with tf.GradientTape() as tape:
                y = tf.nn.conv2d(x, kernel, strides=1, padding='SAME')
                loss = tf.reduce_mean(y * y)
            gradient = tape.gradient(loss, kernel)
            product = tf.linalg.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
            assert 'GPU:0' in y.device and 'GPU:0' in product.device
            tf.debugging.assert_all_finite(gradient, 'Nonfinite convolution gradient')
            print('GPU convolution, gradient and matmul passed:', float(loss.numpy()),
                  float(tf.reduce_sum(product).numpy()))
    else:
        with tf.device('/CPU:0'):
            assert float(tf.reduce_sum(tf.ones((2, 2))).numpy()) == 4.0


if __name__ == '__main__':
    main()
