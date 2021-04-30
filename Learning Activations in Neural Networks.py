#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Building neural network model
#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow import keras
from keras.layers import Dense,Input,LeakyReLU
from keras.models import Sequential
#from keras.callbacks import LambdaCallback
from keras.utils import to_categorical
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def neural_network(data,out_var,in_neurons,hidden_neurons,epoch,batch_size):
    #Splitting the data into input and output
    x = data.drop(out_var,axis=1)
    y = data[out_var]
    
    #splitting the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)
    
    neural_network_model(x_train,x_test,y_train,y_test,in_neurons,hidden_neurons,epoch,batch_size)

#defining neural network model
def neural_network_model(x_train,x_test,y_train,y_test,in_neurons,hidden_neurons,epoch,batch_size): 
    act = ['tanh','relu','softplus','selu','elu','exponential',LeakyReLU(alpha=0.01)]
    df = []
    acc = []
    inp = []
    
    for i in tqdm(range(0,len(act))):
        #creating the model
        model = Sequential()
        #input layer
        model.add(Dense(units=in_neurons,input_dim=x_train.shape[1],kernel_initializer='uniform',activation=act[i],name='input_layer'))
        #1st hidden layer
        model.add(Dense(units=hidden_neurons,kernel_initializer='uniform',activation=act[i],name='hidden_layer'))
        #output layer
        model.add(Dense(units=to_categorical(y_train).shape[1],activation='softmax',name='output_layer'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        #to print the weights after each epoch
        #print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))

        # Fit the model
        history = model.fit(x_train, to_categorical(y_train), validation_data=(x_test,to_categorical(y_test)), epochs=epoch, batch_size=batch_size,verbose=0)
        
        inp.append([model])
        df.append(history)
        acc.append(max(history.history['accuracy']))
    
    #printing summary of trained neural network
    print(model.summary())
    
    res = pd.DataFrame({'Activation_Function':act,'Accuracy':acc})
    res.Activation_Function[6] = 'LeakyRelu'
    print('Best Activation function is {} with accuracy of {}'.format(res.Activation_Function[res.Accuracy.idxmax()],max(res.Accuracy)))
    
    #calculating f1_score
    y_pred = model.predict(x_test)
    print('F1_score for model with AF = {} is {}'.format(res.Activation_Function[res.Accuracy.idxmax()],f1_score(y_test,np.argmax(y_pred,axis=1),average='weighted')))
    
    #creating subplots and fixing the figure size
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
    
    # summarize history for accuracy
    axes[0].plot(df[res.Accuracy.idxmax()].history['accuracy'])
    axes[0].plot(df[res.Accuracy.idxmax()].history['val_accuracy'])
    axes[0].set(xlabel='epoch', ylabel='accuracy')
    axes[0].set_title('model accuracy for AF - {}'.format(res.Activation_Function[res.Accuracy.idxmax()]))
    axes[0].legend(['train', 'test'])

    # summarize history for loss
    axes[1].plot(df[res.Accuracy.idxmax()].history['loss'])
    axes[1].plot(df[res.Accuracy.idxmax()].history['val_loss'])
    axes[1].set(xlabel='epoch', ylabel='loss')
    axes[1].set_title('model loss for AF - {}'.format(res.Activation_Function[res.Accuracy.idxmax()]))
    axes[1].legend(['train', 'test'])
    plt.show()
    
    #printing final epoch weights
    print('Weights of 1st layer after final epoch \n',inp[res.Accuracy.idxmax()][0].layers[0].get_weights()[0],'\n')
    print('Weights of 2nd layer after final epoch \n',inp[res.Accuracy.idxmax()][0].layers[1].get_weights()[0],'\n')
    print('Weights of 3rd layer after final epoch \n',inp[res.Accuracy.idxmax()][0].layers[2].get_weights()[0])
    return res


# In[2]:


#importing the BankNote_Authentication dataset
bank = pd.read_csv('BankNote_Authentication.csv')
bank.head()


# In[3]:


data = bank
out_var = 'class'
input_neurons = 4
hidden_neurons=8
epoch = 20
batch_size = 32
result = neural_network(data,out_var,input_neurons,hidden_neurons,epoch,batch_size)


# In[4]:


#importing cancer dataset
cancer = pd.read_csv('Wisconsin Breast Cancer.csv')
cancer.head()


# In[5]:


#encoding output variable
from sklearn.preprocessing import LabelEncoder
encod = LabelEncoder()
cancer.diagnosis = encod.fit_transform(cancer.diagnosis)
cancer.drop(['id','Unnamed: 32'],axis=1,inplace=True)
cancer.head()


# In[6]:


data = cancer
out_var = 'diagnosis'
input_neurons = 32
hidden_neurons=64
epoch = 20
batch_size = 32
result = neural_network(data,out_var,input_neurons,hidden_neurons,epoch,batch_size)


# In[7]:


#importing iris dataset
iris = pd.read_csv('iris.csv')
iris.head()


# In[8]:


#encoding output variable
from sklearn.preprocessing import LabelEncoder
encod = LabelEncoder()
iris.Species = encod.fit_transform(iris.Species)
iris.head()


# In[9]:


data = iris
out_var = 'Species'
input_neurons = 64
hidden_neurons=300
epoch = 30
batch_size = 32
result = neural_network(data,out_var,input_neurons,hidden_neurons,epoch,batch_size)


# In[12]:


#creating neural network for MNIST data
import mnist

#importing handwritten images
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

#Normalization of images
train_images = train_images/255
test_images = test_images/255

#flatten the images (28*28 = 784)
train_images = train_images.reshape((-1,784))
test_images = test_images.reshape((-1,784))


# In[11]:


x_train = train_images
x_test = test_images
y_train = train_labels
y_test = test_labels
input_neurons = 64
hidden_neurons=128
epoch = 20
batch_size = 32
result = neural_network_model(x_train,x_test,y_train,y_test,input_neurons,hidden_neurons,epoch,batch_size)


# In[ ]:




