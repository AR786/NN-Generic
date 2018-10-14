import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

DATADIR= "C:/Users/prajw/OneDrive/Desktop/fruitsandveggie"

Categories= ["raw","ripe","rotten"]

for category in Categories:
    path = os.path.join(DATADIR,category)  # create path to raw,ripe and rotten
    for img in os.listdir(path):  # iterate over each image per
        img_array = cv2.imread(os.path.join(path,img))
        #cv2.imshow('image',img_array)  



IMG_SIZE =100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))


training_data = []

def create_training_data():
    for category in Categories:  # do 

        path = os.path.join(DATADIR,category)  
        class_num = Categories.index(category)  

        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img) ,) 
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
                training_data.append([new_array, class_num])

            except Exception as e:
                pass


create_training_data()

random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


X=X/255
#from keras.utils import to_categorical

#Z=to_categorical(X)



model = Sequential() 

model.add(Conv2D(256,(3,3),input_shape =X.shape[1:], padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(256,(3,3),padding = 'same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(256,(3,3), padding = 'same'))
model.add(Activation("relu"))

model.add(Conv2D(256,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))

model.add(Dense(3))
model.add(Activation("softmax"))

model.compile(loss = "sparse_categorical_crossentropy",optimizer = "adam",metrics=['accuracy'])

model.fit(X,y,batch_size = 32,epochs = 10,validation_split = 0.2)
model.save('3r-classifier')
new_model = tensorflow.keras.models.load_model('3r-classifier')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import random
from tqdm import tqdm
import tensorflow as tf

df= "C:/Users/prajw/OneDrive/Desktop/fruitsandveggie"

c= ["raw_test","ripe_test","rotten_test"]

for C in c:
    path = os.path.join(df,C)  # create path to raw,ripe and rotten
    for img in os.listdir(path):  # iterate over each image per
        img_array = cv2.imread(os.path.join(path,img))
        #cv2.imshow('image',img_array)  



IMG_SIZE =100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))


test_data = []

def create_test_data():
    for C in c:  # do 

        path = os.path.join(df,C)  
        class_num = c.index(C)  

        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img), ) 
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
                test_data.append([new_array, class_num])

            except Exception as e:
                pass


create_test_data()

random.shuffle(test_data)

X_test = []
y_test = []

for features,label in test_data:
    X_test.append(features)
    y_test.append(label)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
predictions= new_model([X_test])
import numpy as np
print(np.argmax(predictions[0])
