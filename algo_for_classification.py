#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# In[2]:


fashion_mnist = keras.datasets.fashion_mnist
(trainx ,trainy) , (testx,testy) = fashion_mnist.load_data()


# In[4]:


heads = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

trainy


# In[18]:


plt.figure()
plt.imshow(trainx[0])
plt.colorbar()
plt.grid(False)
plt.show()


# In[6]:


trainx = trainx /255.0
testx = testx / 255.0


# In[7]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainx[i], cmap=plt.cm.binary)
    plt.xlabel(heads[trainy[i]])
plt.show()


# In[11]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optamizer ="adma",
              loss = "sparse_categorical_crossentropy",
              metrics=['accuracy'])


# In[12]:


model.fit(trainx ,trainy, epochs = 10)


# In[13]:


test_loss , test_acc = model.evaluate(testx,testy, verbose=2)
print("accuracy = ", test_acc)


# In[14]:


predic = model.predict(testx)
predic[0]


# In[15]:


np.argmax(predic[0])


# In[16]:


testy[0]


# In[45]:


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(heads[predicted_label],
                                100*np.max(predictions_array),
                                heads[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#FF0000")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[58]:


#give any index value to know the acurracry of the modle from given data from 0 to 1000
index_num =int(input("enter any index number :"))
if index_num <10000:
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(index_num, predic[index_num], testy, testx)
    plt.subplot(1,2,2)
    plot_value_array(index_num, predic[index_num],  testy)
    plt.show()
else:
    print("given index number is out of bounds for axis 0 with size 10000")
    


# In[ ]:




