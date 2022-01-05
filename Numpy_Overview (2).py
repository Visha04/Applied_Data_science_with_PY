#!/usr/bin/env python
# coding: utf-8

# # ARRAYS OVERVIEW

# In[2]:


import numpy as np
import math


# In[3]:


a = np.array([1,2,3])


# In[4]:


print(a)


# In[5]:


#print number of dimensions of an array

a.ndim


# In[6]:


#print the data type of content in the array

print(a.dtype)


# In[7]:


#print shape of an array

print(a.shape)


# In[8]:


#creating two dimensional arrays

b = np.array([[1,2,3],[4,5,6]])


# In[9]:


print(b)


# In[10]:


print(b.shape)


# In[11]:


print(b.dtype)


# In[12]:


print(b.ndim) #no of dimesions in an array


# In[13]:


#sometimes we know the shape of the arrays but not what the array will hold;
#So you can create a placeholder and fill it with zeros and ones

d = np.zeros((2,3))
print(d)

e = np.ones((2,3))
print(e)


# In[14]:


#rand;arrange andlinspace
#rand generates random number
#arange to generate a sequence
#linspace togenerate evenly spaced floats

f  = np.random.rand(2,3)
print(f)

g =np.arange(10,50,2)# creates a sequence of numbers between 10 and 50(exclusive) with steps of 2

print(g)

h = np.linspace(3,10,15)# this creates an array of numbers between 3 and 10 and a total of 15 numbers which will be evenly spaced
print(h)


# # ARRAY OPERATIONS

# In[15]:


# ARRAY OPERATIONS APPLY ELEMENT WISE

A = np.array([1,2,3])
B = np.array([4,5,6])

C = A-B
print(C)

D = A+B
print(D)

E = A*B
print(E)

F= A/B
print(F)


# In[16]:


#Example of converting from fahrenheit to celsius

fahrenheit = np.array([0,-10,-5,-15,0])
celsius = (fahrenheit -31)* (5/9)
print(celsius)


# In[17]:


# Boolean arrays:

#Returns True or false arrays based on condition

celsius >-20


# In[18]:


#besides element wise calculation numpy can also do matrix level multiplication
#matrix multiplication is done using @ symbol

a= np.array([[1,2],[4,5]])
b =np.array([[1,2],[4,5]])

print(a@b)


# In[19]:


#concept of upcasting
#when dealing with arrays that have different data types , the resulting array is converted to more general of the two data types:

#For example

array1 = np.array([[1,2,3],[4,5,6]])
array2 =np.array([[1.1,2.2,3.3],[4.4,5.5,6.6]])
print(array1.dtype)
print(array2.dtype)

array3 = array1 * array2

print(array3)
print(array3.dtype)


# In[20]:


#Aggregation functions in numpy arrays 

print(array3.mean())
print(array3.max())
print(array3.min())
print(array3.sum())


# In[21]:


#Exmaple where numpy comes into play in images

#using Python imaging library(PIL)

from PIL import Image
from IPython.display import display


# In[22]:


im = Image.open('/home/jovyan/work/resources/week-1/chris.tiff')


# In[23]:


display(im)


# In[24]:


array = np.array(im)


# In[25]:


print(array)


# In[26]:


array.shape


# In[27]:


mask = np.full(array.shape,255)


# In[28]:


modified_array = mask -array


# In[29]:


print(modified_array)


# In[30]:


modified_array = modified_array.astype(np.uint8)


# In[31]:


display(Image.fromarray(modified_array))


# In[32]:


reshaped = np.reshape(modified_array,(100,400))


# In[33]:


display(Image.fromarray(reshaped))


# # INDEXING,SLICING,INTERATING

# In[34]:


#Indexing and slicing a one dimensional aray is similar to list
ind1 = np.array([1,2,3,4,5])


# In[35]:


print(ind1)


# In[36]:


print(ind1[1:])


# In[37]:


print(ind1[:3])


# In[38]:


print(ind1[4])


# In[40]:


#twodimesnional array

two_d = np.array([[1,2,3],[4,5,6],[7,8,9]])


# In[42]:


print(two_d)


# In[43]:


print(two_d[0:2,1:])


# In[45]:


#boolean array

print(two_d>5)


# In[46]:


#print the values where the conditionis true

print(two_d[two_d>5]) #Linear list


# In[47]:


#Working with a dataset

import os
os.getcwd()


# In[49]:


wines = np.genfromtxt("/home/jovyan/work/resources/datasets/winequality-red.csv")

