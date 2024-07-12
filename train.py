
import matplotlib.pyplot as plt
import seaborn as sns
#import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
import numpy as np
import pandas as pd
import gradio

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150
def get_training_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_arr = cv2.imread(img_path)
            data.append(img_arr)

    data = np.array(data)
    data.dtype = "object"
    return data

train = get_training_data('../input/chest-xray-pneumonia/chest_xray/train')
test = get_training_data('../input/chest-xray-pneumonia/chest_xray/test')
val = get_training_data('../input/chest-xray-pneumonia/chest_xray/val')

x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)
    
for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

positives=[]
negatives=[]
for i in range(len(y_train)):
    if y_train[i]:
        positives.append(x_train[i])
    else:
        negatives.append(x_train[i])

plt.bar(labels, [len(negatives), len(positives)], color=["green", "blue"])
plt.title("Cases count in training data set")
plt.ylabel("count")
plt.show()

plt.imshow(positives[0]) 
plt.title("Pneumonia") 
plt.show()

plt.imshow(negatives[4], cmap="gray") 
plt.title("Normal") 
plt.show()

x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

x_train = x_train.reshape(-1, img_size , img_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

x_test[0].shape

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1) 
y_val = y_val.reshape(-1,1)

datagen = ImageDataGenerator(
    featurewise_center=False, 
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False, 
    zca_whitening=False,
    rotation_range = 30,
    zoom_range = 0.2, 
    width_shift_range=0.1,
    height_shift_range=0.1, 
    horizontal_flip = True, 
    vertical_flip=False)

datagen.fit(x_train)

model = Sequential()
model.add(Conv2D( 32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1))) 
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout (0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3), strides = 1, padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2, padding = 'same'))

model.add(Conv2D(128 , (3,3), strides = 1, padding = 'same' , activation = 'relu')) 
model.add(Dropout (0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' ,  activation = 'relu')) 
model.add(Dropout (0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Flatten())
model.add(Dense(units = 120 , activation = 'relu'))
model.add(Dropout (0.2))
model.add(Dense(units = 1 , activation = 'sigmoid'))
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()
model.compile(optimizer = "rmsprop" , 
              loss =  'binary_crossentropy' ,
              metrics = ['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience = 2,
                                            verbose=1,
                                            factor=0.3,
                                            min_lr=0.000001)

history = model.fit(datagen.flow(x_train,y_train, batch_size = 32),
                    epochs =10 ,
                    validation_data = datagen.flow(x_val, y_val), 
                    callbacks = learning_rate_reduction)

# %%
model.save_weights('kaggle/saved_model_ai/pneumoniadetection')

# %%
print("Loss of the model is -" , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is-" , model.evaluate(x_test,y_test)[1]*100 , "%")

# %%
epochs = list(range(10))
fig, ax = plt.subplots(1,2)
train_acc = history.history ['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history ['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs, train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs, val_acc , 'ro-' , label = 'Validation Accuracy') 
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs") 
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss, 'r-o' , label = 'Validation Loss') 
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()

# %%
predictions = model.predict(x_test)
for i in range(len(predictions)):
    predictions[i] = 1 if predictions[i]>0.5 else 0

# %%
print(classification_report(y_test,predictions,target_names = ['Pneumonia (Class 0)','Normal (Class 1)']))

# %%
cm = confusion_matrix(y_test, predictions)
cm = pd.DataFrame(cm, index = ['0','1'] , columns = ['0', '1'])
cm

# %%
sns.heatmap(cm, cmap='Blues', annot=True, xticklabels = labels , yticklabels = labels) 
plt.show()

# %%
def pneumoniaPrediction(img): 
    img = np.array(img)/255
    img = img.reshape(-1, 150, 150, 1)
    isPneusonic = model.predict(img)[0]
    imgClass = "Normal" if isPneumonic<0.5 else "Pneumonic" 
    return imgClass


pr = model.predict(x_test)
for i in range(len(pr)):
    if pr[i]>8.5: 
        pr[i]=1
    else:
        pr[i]=0





