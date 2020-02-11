import cv2 
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
tf.reset_default_graph()
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

Train_dir = 'Cats_and_Dogs\\train'
Test_dir = 'Cats_and_Dogs\\test'
im_size = 50
LR = 1e-3

model_name = 'dogsVscats- {}-{}.model'.format(LR, '2conv-basic-video') 

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]
    
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(Train_dir)):
        label = label_img(img)
        path = os.path.join(Train_dir, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (im_size,im_size))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(Test_dir)):
        path = os.path.join(Test_dir,img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (im_size,im_size))
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', testing_data)
    return testing_data

def training_model():
    convnet = input_data(shape=[None, im_size, im_size, 1], name='input')
    
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(convnet, tensorboard_dir = 'log')
    return model

def spliting():
    train = train_data[:-500]
    test = train_data[-500:]
    X = np.array([i[0] for i in train]).reshape(-1,im_size,im_size,1)
    Y = [i[1] for i in train]
    
    test_x = np.array([i[0] for i in test]).reshape(-1,im_size,im_size,1)
    test_y = [i[1] for i in test]
    
if __name__ == '__main__':
    img = label_img()
    train_data = create_train_data()
    test_data = process_test_data()
    model  = training_model()
    spliting()
    