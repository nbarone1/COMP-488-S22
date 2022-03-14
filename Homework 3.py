#!/usr/bin/env python
# coding: utf-8

# # Homework 3

# In[70]:


# import statements

import tensorflow as tf
import numpy as np
import time as t
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')
x_train /= 255
y_train /= 255
x_test /= 255
y_test /= 255


# In[71]:


# Set batch size, loss model, trainer, initializer (for the ReLU)
batch_size, lr, num_epochs = 64, 0.0002, 50
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
updater = tf.keras.optimizers.Adam(learning_rate=lr)
initializer = tf.keras.initializers.HeNormal()
r = list(range(1,num_epochs+1))

# set plot function
def pltdynamic(x,y1,y2,ax,colors=['b']):
    ax.plot(x,y1,'b',label="Validation Loss")
    ax.plot(x,y2,'r',label="Training Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()


# In[72]:


# Task 1

lenet = tf.keras.models.Sequential([
    # fist Conv
    tf.keras.layers.Conv2D(filters=12,kernel_size=3,activation='relu',padding='same',input_shape=(28,28,1)),
    #first pool
    tf.keras.layers.AvgPool2D(pool_size=2,strides=2),
    #Dropout
    tf.keras.layers.Dropout(rate=0.5),
    #second Conv
    tf.keras.layers.Conv2D(filters=18,kernel_size=1,activation='relu'),
    #second pool
    tf.keras.layers.AvgPool2D(pool_size=2,strides=2),
    # Dropout
    tf.keras.layers.Dropout(rate=0.5),
    #flatten and dense
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120,activation='relu',kernel_regularizer = tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dense(80,activation='relu',kernel_regularizer = tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dense(10)])


# In[73]:


lenet.compile(optimizer=updater, loss = loss, metrics=['accuracy'])
lenet.summary()


# In[74]:


#Test Function
start_time = t.time()
trylen = lenet.fit(x_train,y_train,batch_size=batch_size,epochs=num_epochs,verbose=0,validation_data=(x_test,y_test))
print("--- %s seconds ---" % (t.time() - start_time))

# Print Scores
testscore = lenet.evaluate(x_test,y_test,verbose=0)
print('Test Score:', testscore[0])
print('Test Score:', testscore[1])
# Print Graph of Loss
vloss = trylen.history['val_loss']
tloss = trylen.history['loss']
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')
pltdynamic(r,vloss,tloss,ax)


# In[ ]:





# In[ ]:




