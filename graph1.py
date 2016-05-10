# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:45:25 2016

@author: suraj
"""

import pandas
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense

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



a = Input(shape=(9,))
b = Dense(2)(a)
model = Model(input=a, output=b)

#model.compile(
#              loss='mean_squared_error',optimizer='sgd',
#              metrics=['accuracy'])
model.compile(
              loss='categorical_crossentropy',optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(X, y,nb_epoch=70,verbose=2) 


preds = model.predict(testX)
print preds.shape
print ("PREDICTED  ","\t","ACTUAL")
for i in range(20):
    print preds[i],"\t",testy[i]


"""
LAST 9 INPUT ATTRIBUTES AS FEATURES
loss='categorical_crossentropy',optimizer='rmsprop'

TEST-SET ACCURACY 15/20

"""
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