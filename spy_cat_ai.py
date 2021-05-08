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
from tensorflow.keras.metrics import categorical_crossentropy
# In[2]:


#img = image.load_img("/home/gabriron01/data/Unfitted/testing/Down/6.tif")


# In[3]:


#plt.imshow(img)


# In[4]:


train = ImageDataGenerator(rescale = 1/255)
validation  = ImageDataGenerator(rescale = 1/255)


# In[5]:


train_dataset = train.flow_from_directory(
    'spy_2hrs/data/order_408684/SPY_Plots/Fitted_2_hours/training/',
    target_size = (469, 469),
    batch_size = 20,
    class_mode = 'categorical'
)
validation_dataset = validation.flow_from_directory(
    'spy_2hrs/data/order_408684/SPY_Plots/Fitted_2_hours/validation/',    
    target_size = (469,469),
    batch_size = 20,
    class_mode  = 'categorical'
)

print('training classes index: {}'.format(train_dataset.class_indices))
print('validation classes index: {}'.format(validation_dataset.class_indices))
#sys.exit(0)

# In[6]:


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3), padding='same', activation = 'relu',input_shape = (469,469,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                   #
                                    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(128,(3,3), padding='same', activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    #tf.keras.layers.Conv2D(128,(3,3),activation = 'relu'),
                                    #tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Flatten(),
                                                           ##
                                    tf.keras.layers.Dense(512,activation = 'relu'),
                                                           ##
                                    tf.keras.layers.Dense(3, activation = 'softmax')
                                   ])


# In[7]:


model.compile(loss = 'categorical_crossentropy',
             optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])


# In[ ]:


model_fit = model.fit(train_dataset,
                     steps_per_epoch = 15,
                     epochs = 40,
                     validation_data = validation_dataset)


# In[ ]:



saved_mode = model.save('my_model')

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
