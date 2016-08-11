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

```
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

```
## Statistics and Sample Output


```
Epoch 1/7
99950/99950 [==============================] - 385s - loss: 2.6995
Epoch 2/7
99950/99950 [==============================] - 405s - loss: 2.2101
Epoch 3/7
99950/99950 [==============================] - 392s - loss: 2.0294
Epoch 4/7
99950/99950 [==============================] - 416s - loss: 1.9114
Epoch 5/7
99950/99950 [==============================] - 409s - loss: 1.8369
Epoch 6/7
99950/99940 [==============================] - 392s - loss: 1.6254
Epoch 7/7
99950/99950 [==============================] - 407 - loss: 1.5024

```


## Sample Output 1

```
TRANIO:
You will be schoolmaster
And undertake the teaching of the maid:
That's your device.

LUCENTIO:
It is: may it be done?

TRANIO:
Not possible; for who shall bear your part,
And be in Padua here Vincentio's son,
Keep house and ply his book, welcome his friends,
Visit his countrymen and banquet them?

```
## Sample Output 2

```
If one have understood how "Sin came into the
world," namely through errors of the reason, through which men in their
intercourse with one another and even individual men looked upon
themselves as much blacker and wickeder than was really the case, one's
whole feeling is much lightened and man and the world appear together in
such a halo of harmlessness that a sentiment of well being is instilled
into one's whole nature. Man in the midst of nature is as a child left
to its own devices. This child indeed dreams a heavy, anxious dream. But
when it opens its eyes it finds itself always in paradise.


125

=Irreligiousness of Artists.=--Homer is so much at home among his gods
and is as a poet so good natured to them that he must have been
profoundly irreligious. That which was brought to him by the popular
faith--a mean, crude and partially repulsive superstition--he dealt with
as freely as the Sculptor with his clay, therefore with the same freedom
that Ã†schylus and Aristophanes evinced and with which in later times the
great artists of the renaissance, and also Shakespeare and Goethe, drew
their pictures.

```
