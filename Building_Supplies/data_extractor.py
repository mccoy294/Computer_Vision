import tensorflow as tf
import os

#Avoid OOM errors by setting aside GPU Memory Constraints
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
