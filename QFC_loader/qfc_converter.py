
# coding: utf-8

# THis is a QFC data convert, load all jpg recursively under the folder
# convert all into x_train, y_train, x_test, y_test
# where 
# x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, 32, 32).
# y_train, y_test: uint8 array of category labels (integers in range 0,1,2,3) with shape (num_samples,).
# 
# RAW FILES:
# files name 
# 
# LF: Left Front 0
# 
# LR: Left Rear 1
# 
# RF: Right Front 2
# 
# RR: Right Rear 3
# 
# find all figures recursively, crop the middle square, downsize to 224*224, save all into train files

# In[1]:

DATA_PATH = 'data/'
test_ratio = 0.1


# In[2]:

import fnmatch
import os
import numpy
from keras.preprocessing import image

matches = []
target = []

for root, dirs, files in os.walk(DATA_PATH):
    for filename in files:
        if filename.endswith(('.jpg', '.jpeg', '.png','.JPG', '.JPEG', '.PNG')):
            matches.append(os.path.join(root, filename))
            if "LF." in filename:
                target.append([1,0,0,0])
            if "LR." in filename:
                target.append([0,1,0,0])
            if "RF." in filename:
                target.append([0,0,1,0])
            if "RR." in filename:
                target.append([0,0,0,1])
            


# In[3]:

def cropcenter(img):
    width, height = img.size
    if width <= height:
        length = width
    else:
        length = height

    left = (width - length)/2
    top = (height - length)/2
    right = (width + length)/2
    bottom = (height + length)/2

    img = img.crop((left,top,right,bottom))
    img = img.resize((224,224))
    return img


# In[4]:

print(matches)
print(target)


# In[5]:

all_files_count = len(target)
test_files_count = int(all_files_count*test_ratio)
train_files_count = all_files_count - test_files_count


# In[6]:

istest = numpy.zeros(all_files_count)
istest[:test_files_count] = 1
numpy.random.shuffle(istest)
istest


# In[7]:

img_path = matches[2]
#img = image.load_img(img_path, target_size=(224, 224))
img = image.load_img(img_path)
img = cropcenter(img)
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
img


# In[8]:

x_train = []
x_test = []
y_train = []
y_test = []


# In[9]:

for i in matches:
    index = matches.index(i)
    img_path = matches[index]
    img = image.load_img(img_path)
    arr = image.img_to_array(cropcenter(img)).astype("uint8")
    if istest[index]:
        x_test.append(arr)
        y_test.append(target[index])
    else:
        x_train.append(arr)
        y_train.append(target[index])        
x_test=numpy.array(x_test)
x_train=numpy.array(x_train)
y_test=numpy.array(y_test)
y_train=numpy.array(y_train)


# In[10]:

numpy.save("qfc_x_train.npy", x_train)
numpy.save("qfc_x_test.npy", x_test)
numpy.save("qfc_y_train.npy", y_train)
numpy.save("qfc_y_test.npy", y_test)


# In[ ]:




# In[ ]:




# In[ ]:



