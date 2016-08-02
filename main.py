# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 08:40:19 2016

@author: Suraj Jayakumar
"""

import numpy as np

from keras.models import Model,model_from_json, Sequential
from keras.layers import Input, Dense, Dropout, LSTM
from keras.utils.data_utils import get_file


##################################
##### GLOBAL VARIABLES ###########
X_train =0
X_test = 0
Y_train = 0
Y_test = 0
model = 0


# data I/O
path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
data= open(path).read().lower()

#data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }


train_samples = int(data_size*0.8)

def create_data():
    global X_train,X_test,Y_train,Y_test


    X = np.zeros((data_size,1,vocab_size))
    y = np.zeros((data_size,vocab_size))


    for i in xrange(data_size):
        this_char = data[i]
        next_char = data[(i+1)%data_size]
        X[i][0][char_to_ix[this_char]] = 1.
        y[i][char_to_ix[next_char]] = 1.

    X_train = X
    Y_train = y
    X_test = X_train
    Y_test = Y_train


def build_model():

    global model

    input_layer = Input(shape=(1,vocab_size))

    lstm_1 = LSTM(128,return_sequences=True)(input_layer)

    lstm_2 = LSTM(64)(lstm_1)

    output_layer = Dense(vocab_size,activation='softmax')(lstm_2)

    model = Model(input=input_layer,output=output_layer)

#    print('Build model...')
#    model = Sequential()
#    model.add(LSTM(128, input_shape=(1, vocab_size)))
#    model.add(Dense(vocab_size,activation='softmax'))


    model.summary()
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


def train_model():
    #verbose = 1 for progress bar logging, 2 for one log line per epoch.
    global model
    model.fit(X_train, Y_train,
                    batch_size=1, nb_epoch=10,
                    verbose=2)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print score


def save_model():
    global model
    open('lan_model.json','w').write(model.to_json())
    model.save_weights('lan_model.h5',overwrite=True)

def load_model():
    global model
    model = model_from_json(open('lan_model.json').read())
    model.load_weights('lan_model.h5')
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


def print_model_output(start,end):

    final_str=""

if __name__=="__main__":
    create_data()
    build_model()
    train_model()
    save_model()
