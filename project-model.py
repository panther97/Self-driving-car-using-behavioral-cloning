#!/usr/bin/env python
# coding: utf-8

# In[3]:


import keras
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


# In[4]:


datadir = '/kaggle/input/drive-dataset/track-master'
coloumns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir,'driving_log.csv'), names = coloumns)


# In[5]:


def name_split(name):
    h, t = ntpath.split(name)
    return t
data['center'] = data['center'].apply(name_split)
data['left'] = data['left'].apply(name_split)
data['right'] = data['right'].apply(name_split)
data.head()


# In[6]:


bins = 25
epsilon = 400
hist, bins = np.histogram(data['steering'], bins)
center = bins[:-1] + bins[1:]
center*=0.5


# In[7]:


remove = []
for i in range(25):
    l = []
    for j in range(len(data['steering'])):
        if(data['steering'][j] >= bins[i] and data['steering'][j] <=bins[i+1] ):
            l.append(j)
    l = shuffle(l)
    l = l[epsilon:]
    remove.extend(l)


# In[8]:


print(data.iloc[0])
def load_images(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        index = data.iloc[i]
        center, left, right = index[0], index[1], index[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(index[3]))
    image_path = np.asarray(image_path)
    steering = np.asarray(steering)
    return image_path, steering
image_path, steering = load_images(datadir + '/IMG', data)
        


# In[9]:


X_train, X_test, Y_train, Y_test = train_test_split(image_path, steering, test_size = 0.2, random_state = 6)


# In[10]:


def zoom(image):     #Applies Zoom augmentation to image
    zoom = iaa.Affine(scale = (1,1.3))
    image = zoom.augment_image(image)
    return image


# In[13]:


def pan(image):     #Applies lateral shift augmentation to image
    pan = iaa.Affine(translate_percent = {"x": (-0.1,0.1), "y": (-0.1,0.1)})
    image = pan.augment_image(image)
    return image


# In[14]:


def bright(image):    #Applies brightness augmentation to image
    bright = iaa.Multiply((0.2,1.2))
    image = bright.augment_image(image)
    return image


# In[15]:


def flip(image, steering_angle):    #flips the image horizontally and changes the sign of steering angle
    image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return image, steering_angle


# In[16]:


def warp(image):        #Applies perspective transform to image
    w = iaa.PerspectiveTransform(scale=(0.01, 0.15))
    image = w.augment_image(image)
    return image


# In[17]:


def random_augment(image, steering_angle):    #Randomly applies augmentation to images with 50% probablity
    img = mpimg.imread(image)
    if(np.random.rand()<0.5):
        img = pan(img)
    if(np.random.rand()<0.5):
        img, steering_angle = flip(img, steering_angle)
    if(np.random.rand()<0.5):
        img = bright(img)
    if(np.random.rand()<0.5):
        img = zoom(img)
    if(np.random.rand()<0.5):
        img = warp(img)
    return img, steering_angle
        


# In[18]:


def preprocess(img):
    img = img[60:137]        #cropping the image to remove extra information
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)    #converting RGB image to YUV
    img = cv2.GaussianBlur(img, (3,3),0)    #Applying Guassian Blur to remove unneccessary noise
    img = cv2.resize(img, (200,66))      #Resizing image to the input size of image required by Nvidia Model
    img = img/255  #Normalizing the image
    return img


# In[19]:


img = image_path[100]
original  = mpimg.imread(img)
preprocessed = preprocess(original)


# In[20]:


def batch_generator(image_path, steering, batch_size, is_training):
    while True:
        batch_image = []
        batch_steering = []
        
        for i in range(batch_size):
            random_index = random.randint(0, len(image_path) -1)
            
            if(is_training ==True):
                image, steer = random_augment(image_path[random_index], steering[random_index])
            else:
                image = mpimg.imread(image_path[random_index])
                steer = steering[random_index]
            image = preprocess(image)
            batch_image.append(image)
            batch_steering.append(steer)
        yield(np.asarray(batch_image), np.asarray(batch_steering))
    


# In[21]:


X_train_gen, Y_train_gen = next(batch_generator(X_train, Y_train, 1,1))
X_test_gen, Y_test_gen = next(batch_generator(X_test, Y_test, 1,0))


# In[22]:


def nvidia_model():
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
  
    model.add(Convolution2D(64, 3, 3, activation='elu'))
  
    model.add(Flatten())
  
    model.add(Dense(100, activation = 'elu'))
  
    model.add(Dense(50, activation = 'elu'))
      
    model.add(Dense(10, activation = 'elu'))
 
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer)
    return model


# In[23]:


model = nvidia_model()
print(model.summary())


# In[24]:


history = model.fit_generator(batch_generator(X_train, Y_train, 100, 1), 
                              steps_per_epoch =300,
                              epochs = 13,
                              validation_data = batch_generator(X_test, Y_test,100, 0),
                              validation_steps = 200,
                              verbose = 1,
                              shuffle = 1)


# In[25]:


plt.plot(history.history['loss'], label = "Training Loss")
plt.plot(history.history['val_loss'], label = "Testing Loss")
plt.title("Training Loss vs Testing Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")


# In[26]:


model.save('model.h5')


# In[ ]:




