# language-modeling


## Getting and Cleaning Data

Data has 600901 characters, 59 unique.


Each letter is represented using a **One Hot Vector** of the size of the total vocabulary.

Example:
Suppose there are 59 unique chars in the vocabulary

'c' => `a =   [0,0,0,0,1,0,0,0,0,0,,....]`
where a[4] = 1 represents the character 'c'

Number of training samples: 500000
Number of testing samples: 100000

X : (60000L, 28L, 28L)
y : (60000L, 10L)



## Proposed Neural Network Architecture

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
InputLayer			             (None, 1, 59)         0
____________________________________________________________________________________________________
LSTM 1		                     (None, 1, 128)        96256       InputLayer
____________________________________________________________________________________________________
LSTM 2		                     (None, 64)            49408       LSTM 1
____________________________________________________________________________________________________
Dense 		                     (None, 59)            3835        LSTM 2
====================================================================================================
Total params: 149499
____________________________________________________________________________________________________


## Statistics and Sample Output