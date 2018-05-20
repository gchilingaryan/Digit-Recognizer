import pandas as pd
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import *
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam

MODEL_NAME = "digit-recognizer.h5"

df = pd.read_csv('train.csv')
train_df = df.iloc[:,1:].values
test_df = pd.read_csv('test.csv').values
pixel_label = df['label'].values

train_df = np.array(train_df).reshape((-1, 1, 28, 28)).astype(np.uint8)
test_df = np.array(test_df).reshape((-1, 1, 28, 28)).astype(np.uint8)
pixel_label = to_categorical(pixel_label, 10).astype(np.uint8)


def training_data():
    training_data = []
    for i, data in tqdm(enumerate(train_df)):
        label = pixel_label[i]
        training_data.append([np.array(data), np.array(label)])
    shuffle(training_data)
    np.save("training_data.npy", training_data)
    return training_data

def testing_data():
    testing_data = []
    for i, data in tqdm(enumerate(test_df)):
        testing_data.append([np.array(data), i+1])
    np.save("testing_data.npy", testing_data)
    return testing_data

# For creating the numpy file
# training_data = training_data()
# testing_data = testing_data()

#  For loading the numpy file
training_data = np.load("training_data.npy")
testing_data = np.load("testing_data.npy")

# #######training#########

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(10, activation='softmax'))
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=optimizer, loss='categorical_crossentropy' ,metrics=['accuracy'])

if os.path.exists(MODEL_NAME):
    model.load_weights(MODEL_NAME)
    print 'model exists'

train = training_data[:-500]
test = training_data[-500:]
X = np.array([i[0] for i in train]).reshape([-1, 28, 28, 1])
y= np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape([-1, 28, 28, 1])
test_y = np.array([i[1] for i in test])

model.fit(X, y, epochs=5, verbose=1, validation_data=(test_x, test_y))

model.save(MODEL_NAME)

######plotting######
fig = plt.figure()
for i, data in enumerate(testing_data[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3,4,i+1)
    data = img_data.reshape(-1,28,28,1)
    model_out = model.predict([data])[0]

    label = np.argmax(model_out)

    y.imshow(img_data[0], cmap=cm.binary)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    plt.title(label)
plt.show()


















