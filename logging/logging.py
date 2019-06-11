import tensorflow as tf

def log_scalar(name, value):
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar(name, value)