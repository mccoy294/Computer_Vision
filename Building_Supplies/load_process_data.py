import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Generate the data using keras utilities
data = tf.keras.utils.image_dataset_from_directory("data")

#Pull out the data from the generator into a numpy array
data_iterator = data.as_numpy_iterator()


#Class 0 = roofing
#Class 1 = Sheetrock

#create a batch
batch = data_iterator.next()



#Pre-processing steps____________________

scaled = batch[0] / 255

print(scaled.max())

