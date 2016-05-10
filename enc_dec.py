# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:58:30 2016

@author: suraj
"""

from keras.layers import Input, Embedding, LSTM, Dense, merge,GRU
from keras.models import Model
import pandas
import numpy as np






mydata =pandas.read_csv("mydata.csv")
finaldata = np.array(mydata)
np.random.seed = 1337
np.random.shuffle(finaldata)
X = finaldata[:-20,49:58]/16000.
#y = finaldata[:-20,58]
testX = finaldata[-20:,49:58]/16000.
#testy = finaldata[-20:,58]
newy = np.zeros(shape=(len(finaldata),2))
for i in range(len(finaldata)):
    if(finaldata[i][-1]==0):
        newy[i][0] = 1
    elif(finaldata[i][-1]==1):
        newy[i][1] = 1
y = newy[:-20]
testy = newy[-20:]

print X.shape
print y.shape
print testX.shape
print testy.shape

'''
newy is one-hot-encoded output
'''



main_input = Input(shape=(9,1), dtype='float32', name='main_input')
lstm_encoder = LSTM(18,return_sequences=True)(main_input)
interim_output = Dense(1,)(lstm_encoder)
encoder_model = Model(input=main_input, output=interim_output)

lstm_decoder = LSTM(encoder_model)(interim_output)
final_output = Dense(2, activation='softmax')(lstm_decoder)
decoder_model = Model(input=lstm_decoder, output=interim_output)

model.compile(
              loss='categorical_crossentropy',optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(np.reshape(X,newshape=(len(X),9,1)), y,nb_epoch=1,verbose=2) 
preds = model.predict(np.reshape(testX,newshape=(len(testX),9,1)))
print preds.shape
print ("PREDICTED  ","\t","ACTUAL")
for i in range(20):
    print preds[i],"\t",testy[i]
    
for i in range(20):
    temp_pred =0
    temp_actual=0
    if(preds[i][0]>preds[i][1]):
        temp_pred=0
    elif(preds[i][0]<preds[i][1]):
        temp_pred=1
    if(testy[i][0]>testy[i][1]):
        temp_actual=0
    elif(testy[i][0]<testy[i][1]):
        temp_actual=1
    print temp_pred,temp_actual
    
    
    
    
'''

main_input = Input(shape=(9,1), dtype='float32', name='main_input')
#Let 32 be output dim of LSTM Encoder
lstm_encoder = LSTM(18,return_sequences=True)(main_input)
#We pass the output and activations learnt from LSTM_Encoder along with the main_input

lstm_decoder = LSTM(9)(lstm_encoder)
#OR
#lstm_decoder = GRU(lstm_encoder)(main_input)
output = Dense(2, activation='softmax')(lstm_decoder)
model = Model(input=main_input, output=output)

model.compile(
              loss='categorical_crossentropy',optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(np.reshape(X,newshape=(len(X),9,1)), y,nb_epoch=1,verbose=2) 
'''