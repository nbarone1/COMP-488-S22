#!/usr/bin/env python
# coding: utf-8

# In[12]:


#  Homework 2

# Import Statements
import tensorflow as tf
from d2l import tensorflow as d2l
import numpy as np
import time as t
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set batch size, loss model, trainer, initializer (for the ReLU)
batch_size, lr, num_epochs = 32, 0.0002, 50
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
updater = tf.keras.optimizers.Adam(learning_rate=lr)
initializer = tf.keras.initializers.HeNormal()
r = list(range(1,num_epochs+1))


# In[13]:


# set plot function
def pltdynamic(x,y1,y2,ax,colors=['b']):
    ax.plot(x,y1,'b',label="Validation Loss")
    ax.plot(x,y2,'r',label="Training Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw


# In[14]:


# Data set up

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')
x_train /= 255
y_train /= 255
x_test /= 255
y_test /= 255


# In[15]:


# MLPS Task 1


# In[16]:


# 2 Layer

n2layer = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    # hidden layer 1
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer),
    # hidden layer 2
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer),
    tf.keras.layers.Dense(10)])
n2layer.compile(optimizer=updater, loss = loss, metrics=['accuracy'])

#Test Function
start_time = t.time()
trymlp2 = n2layer.fit(x_train,y_train,batch_size = batch_size,epochs = num_epochs, verbose=0, validation_data=(x_test,y_test))
print("--- %s seconds ---" % (t.time() - start_time))

# Print Scores
testscore2 = n2layer.evaluate(x_test,y_test,verbose=0)
print('Test Score:', testscore2[0])
print('Test Score:', testscore2[1])
# Print Graph of Loss
vloss2 = trymlp2.history['val_loss']
tloss2 = trymlp2.history['loss']
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')
pltdynamic(r,vloss2,tloss2,ax)


# In[17]:


# 3 Layer

n3layer = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    # hidden layer 1
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer),
    # hidden layer 2
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer),
    # hidden layer 3
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer),
    tf.keras.layers.Dense(10)])
n3layer.compile(optimizer=updater, loss = loss, metrics=['accuracy'])

#Test Function
start_time = t.time()
trymlp3 = n3layer.fit(x_train,y_train,batch_size = batch_size,epochs = num_epochs, verbose=0, validation_data=(x_test,y_test))
print("--- %s seconds ---" % (t.time() - start_time))

# Print Scores
testscore3 = n3layer.evaluate(x_test,y_test,verbose=0)
print('Test Score:', testscore3[0])
print('Test Score:', testscore3[1])
# Print Graph of Loss
vloss3 = trymlp3.history['val_loss']
tloss3 = trymlp3.history['loss']
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')
pltdynamic(r,vloss3,tloss3,ax)


# In[18]:


# 4 Layer

n4layer = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    # hidden layer 1
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer),
    # hidden layer 2
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer),
    # hidden layer 3
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer),
    # hidden layer 4
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer),
    tf.keras.layers.Dense(10)])
n4layer.compile(optimizer=updater, loss = loss, metrics=['accuracy'])

#Test Function
start_time = t.time()
trymlp4 = n4layer.fit(x_train,y_train,batch_size = batch_size,epochs = num_epochs, verbose=0, validation_data=(x_test,y_test))
print("--- %s seconds ---" % (t.time() - start_time))

# Print Scores
testscore4 = n4layer.evaluate(x_test,y_test,verbose=0)
print('Test Score:', testscore4[0])
print('Test Score:', testscore4[1])
# Print Graph of Loss
vloss4 = trymlp4.history['val_loss']
tloss4 = trymlp4.history['loss']
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')
pltdynamic(r,vloss4,tloss4,ax)


# In[ ]:


### The cost for adding the 3rd and 4th layer, although do improve the convergence rate, they cost more
### in terms of resources and 2 layer appears to be sufficient for the MSINT 100 unit hidden layer model


# In[23]:


## Task 2

# New weight initialization method

initializer2 = tf.keras.initializers.RandomNormal()
initializer3 = tf.keras.initializers.GlorotNormal()

# New Regularizations - Use 'l1' and 'l2'

# New Optimizers

optimizer2 = tf.keras.optimizers.RMSprop(learning_rate=lr)
optimizer3 = tf.keras.optimizers.SGD(learning_rate=lr)


# In[24]:


# 2 Layer Tests using optimizer/initializer/regularization 2 and then 3


# In[31]:


# 2 Layer 2nd Choice

net2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    # hidden layer 1
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer2, kernel_regularizer = tf.keras.regularizers.L1(0.01)),
    # hidden layer 2
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer2, kernel_regularizer = tf.keras.regularizers.L1(0.01)),
    tf.keras.layers.Dense(10)])
net2.compile(optimizer=optimizer2, loss = loss, metrics=['accuracy'])

#Test Function
start_time = t.time()
trymlp22 = net2.fit(x_train,y_train,batch_size = batch_size,epochs = num_epochs, verbose=0, validation_data=(x_test,y_test))
print("--- %s seconds ---" % (t.time() - start_time))

# Print Scores
testscore22 = net2.evaluate(x_test,y_test,verbose=0)
print('Test Score:', testscore22[0])
print('Test Score:', testscore22[1])
# Print Graph of Loss
vloss22 = trymlp22.history['val_loss']
tloss22 = trymlp22.history['loss']
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')
pltdynamic(r,vloss22,tloss22,ax)


# In[33]:


# 2 Layer 3rd Choice

net3 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    # hidden layer 1
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer3, kernel_regularizer = tf.keras.regularizers.L2(0.01)),
    # hidden layer 2
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer3, kernel_regularizer = tf.keras.regularizers.L2(0.01)),
    tf.keras.layers.Dense(10)])
net3.compile(optimizer=optimizer3, loss = loss, metrics=['accuracy'])

#Test Function
start_time = t.time()
trymlp23 = net3.fit(x_train,y_train,batch_size = batch_size,epochs = num_epochs, verbose=0, validation_data=(x_test,y_test))
print("--- %s seconds ---" % (t.time() - start_time))

# Print Scores
testscore23 = net3.evaluate(x_test,y_test,verbose=0)
print('Test Score:', testscore23[0])
print('Test Score:', testscore23[1])
# Print Graph of Loss
vloss23 = trymlp23.history['val_loss']
tloss23 = trymlp23.history['loss']
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')
pltdynamic(r,vloss23,tloss23,ax)


# In[34]:


### The HeNormal() initializer and Adam Optimizer with no regulizer operates the most efficient and effectively
### The combo of L1, RandomNormal and RMSprop operates better than the L2, GlorotNormal, and SGD combination


# In[ ]:




