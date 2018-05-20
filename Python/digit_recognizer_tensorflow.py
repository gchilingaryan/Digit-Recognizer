import pandas as pd
import numpy as np
import os
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import *
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

MODEL_NAME = "digit-recognizer.model"

df = pd.read_csv('train.csv')
train_df = df.iloc[:,1:].values
test_df = pd.read_csv('test.csv').values
pixel_label = df['label'].values

train_df = np.array(train_df).reshape((-1, 1, 28, 28)).astype(np.uint8)
test_df = np.array(test_df).reshape((-1, 1, 28, 28)).astype(np.uint8)
pixel_label = pixel_label.astype(np.uint8)

def label_img(label):
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if label == 0:
        labels[0] = 1
    elif label == 1:
        labels[1] = 1
    elif label == 2:
        labels[2] = 1
    elif label == 3:
        labels[3] = 1
    elif label == 4:
        labels[4] = 1
    elif label == 5:
        labels[5] = 1
    elif label == 6:
        labels[6] = 1
    elif label == 7:
        labels[7] = 1
    elif label == 8:
        labels[8] = 1
    elif label == 9:
        labels[9] = 1

    return labels

def training_data():
    training_data = []
    for i, data in tqdm(enumerate(train_df)):
        label = label_img(pixel_label[i])
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

tf.reset_default_graph()

convnet = input_data(shape=[None, 28, 28, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print 'model exists'

train = training_data[:-500]
test = training_data[-500:]
X = np.array([i[0] for i in train]).reshape([-1, 28, 28, 1])
y= np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape([-1, 28, 28, 1])
test_y = np.array([i[1] for i in test])

model.fit({'input': X}, {'targets': y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

#######plotting######
fig = plt.figure()
for i, data in enumerate(testing_data[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3,4,i+1)
    data = img_data.reshape(28,28,1)
    model_out = model.predict([data])[0]

    label = np.argmax(model_out)

    y.imshow(img_data[0], cmap=cm.binary)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    plt.title(label)
plt.show()



















