#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#import cv2
import os
#from IPython.display import display
#from PIL import Image
import sys
# In[2]:


#img = image.load_img("/home/gabriron01/data/Unfitted/testing/Down/6.tif")


# In[3]:


#plt.imshow(img)


# In[4]:


train = ImageDataGenerator(rescale = 1/255)
validation  = ImageDataGenerator(rescale = 1/255)


# In[5]:


train_dataset = train.flow_from_directory(
    '2hrs_python_generated/data/Aws_Plots/Python/training',
    target_size = (800 ,800),
    batch_size = 20,
    class_mode = 'binary'
)
validation_dataset = validation.flow_from_directory(
    '2hrs_python_generated/data/Aws_Plots/Python/validation',
    target_size = (800 ,800),
    batch_size = 20,
    class_mode  = 'binary'
)

print('training classes index: {}'.format(train_dataset.class_indices))
print('validation classes index: {}'.format(validation_dataset.class_indices))
#sys.exit(0)

# In[6]:


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (800,800,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                   #
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    #tf.keras.layers.Conv2D(128,(3,3),activation = 'relu'),
                                    #tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Flatten(),
                                                           ##
                                    tf.keras.layers.Dense(512,activation = 'relu'),
                                                           ##
                                    tf.keras.layers.Dense(1,activation = 'sigmoid')
                                   ])


# In[7]:


model.compile(loss = 'binary_crossentropy',
             optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])


# In[ ]:


model_fit = model.fit(train_dataset,
                     steps_per_epoch = 15,
                     epochs = 15,
                     validation_data = validation_dataset)


# In[ ]:



# saved_mode = model.save('superimposed_model')
saved_mode = model.save('my_model_python_img')

"""
img = image.load_img('Unfitted/testing/Up/1006.tif', target_size=(800, 800))
X = image.img_to_array(img)
X = np.expand_dims(X, axis=0)
images = np.vstack([X])
val = model.predict(images, verbose=1)
print('prediction: {}'.format(val))
print('classes: {}'.format(np.argmax(val, axis=1)))
#print('classes: {}'.format(train_dataset.classes))
"""
