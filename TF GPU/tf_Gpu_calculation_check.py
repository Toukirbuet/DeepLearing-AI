import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()

# tf.config.experimental.list_physical_devices('GPU') 
# tf.debugging.set_log_device_placement(True)

# Create some tensors

def cpu():
  with tf.device('/cpu:0'):
      tf.random.set_seed(5)
      a = tf.random.normal([2000,2000], 0, 1, tf.float32, seed=1)
      b = tf.random.normal([2000,2000], 0, 1, tf.float32, seed=1)
      c = tf.matmul(a, b)
      return c
def gpu():
  with tf.device('/device:GPU:0'):
      tf.random.set_seed(5)
      a = tf.random.normal([2000,2000], 0, 1, tf.float32, seed=1)
      b = tf.random.normal([2000,2000], 0, 1, tf.float32, seed=1)
      c = tf.matmul(a, b)
      return c


print('CPU (s):')
print(cpu())
cpu_time = timeit.timeit('cpu()', number=1, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
print(gpu())
gpu_time = timeit.timeit('gpu()', number=1, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))