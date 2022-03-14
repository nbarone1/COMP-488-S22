#!/usr/bin/env python
# coding: utf-8

# # Homework 1

# ## Task 1

# In[1]:


# Import Statements
import tensorflow as tf
from d2l import tensorflow as d2l
import numpy as np
import time as t
import matplotlib.pyplot as plt
import sklearn.model_selection as skl

# Set batch size, loss model, trainer, initializer (for the ReLU)
batch_size, lr, num_epochs = 32, 0.0001, 50
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
updater = tf.keras.optimizers.SGD(learning_rate=lr)
initializer = tf.keras.initializers.HeNormal()
r = list(range(1,num_epochs+1))

# set plot function
def pltdynamic(x,y1,y2,ax,colors=['b']):
    ax.plot(x,y1,'b',label="Validation Loss")
    ax.plot(x,y2,'r',label="Training Loss")
    plt.legend()
    plt.grid()
    fig.canvas.draw()


# In[2]:


# Data set up
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

X = np.concatenate([x_train, x_test])
Y = np.concatenate([y_train, y_test])

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')
x_train /= 255
y_train /= 255
x_test /= 255
y_test /= 255


# In[5]:


# Define MLP
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    # hidden layer 1
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer),
    # hidden layer 2
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer),
    tf.keras.layers.Dense(10)])
net.compile(optimizer=updater, loss = loss, metrics=['accuracy'])


# In[4]:


#Test Function
start_time = t.time()
tryReLU = net.fit(x_train,y_train,batch_size = batch_size,epochs = num_epochs, verbose=0, validation_data=(x_test,y_test))
print("--- %s seconds ---" % (t.time() - start_time))

# Print Scores
testscore = net.evaluate(x_test,y_test,verbose=0)
print('Test Score:', testscore[0])
print('Test Score:', testscore[1])
# Print Graph of Loss
vloss = tryReLU.history['val_loss']
tloss = tryReLU.history['loss']
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')
pltdynamic(r,vloss,tloss,ax)


# ## Task 2

# In[110]:


# new initializer (Xavier/Glorot)
initializer_xavier = tf.keras.initializers.GlorotNormal()


# In[111]:


# tanh
# Define MLP
nett = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    # hidden layer 1
    tf.keras.layers.Dense(100, activation='tanh', kernel_initializer=initializer_xavier),
    # hidden layer 2
    tf.keras.layers.Dense(100, activation='tanh', kernel_initializer=initializer_xavier),
    tf.keras.layers.Dense(10)])
nett.compile(optimizer=updater, loss = loss, metrics=['accuracy'])

start_time = t.time()
tryTanH = nett.fit(x_train,y_train,batch_size = batch_size,epochs = num_epochs, verbose=0, validation_data=(x_test,y_test))
print("--- %s seconds ---" % (t.time() - start_time))

# Print Scores
testscore = nett.evaluate(x_test,y_test,verbose=0)
print('Test Score:', testscore[0])
print('Test Score:', testscore[1])
# Print Graph of Loss
vloss = tryTanH.history['val_loss']
tloss = tryTanH.history['loss']
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')
pltdynamic(r,vloss,tloss,ax)


# In[112]:


# sigmoid
# Define MLP
nets = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    # hidden layer 1
    tf.keras.layers.Dense(100, activation='sigmoid', kernel_initializer=initializer_xavier),
    # hidden layer 2
    tf.keras.layers.Dense(100, activation='sigmoid', kernel_initializer=initializer_xavier),
    tf.keras.layers.Dense(10)])
nets.compile(optimizer=updater, loss = loss, metrics=['accuracy'])

start_time = t.time()
trySig = nets.fit(x_train,y_train,batch_size = batch_size,epochs = num_epochs, verbose=0, validation_data=(x_test,y_test))
print("--- %s seconds ---" % (t.time() - start_time))

# Print Scores
testscore = nets.evaluate(x_test,y_test,verbose=0)
print('Test Score:', testscore[0])
print('Test Score:', testscore[1])
# Print Graph of Loss
vloss = trySig.history['val_loss']
tloss = trySig.history['loss']
fig,ax = plt.subplots(1,1)
ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')
pltdynamic(r,vloss,tloss,ax)

