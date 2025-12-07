import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Simple test to make sure TensorFlow runs properly
hello = tf.constant("Hello, TensorFlow!")
tf.print(hello)