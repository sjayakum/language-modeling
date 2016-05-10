

## Encoder Decoder Architecture

### Type 1

`
main_input = Input(shape=(9,1), dtype='float32', name='main_input')
lstm_encoder = LSTM(200,return_sequences=True)(main_input)
lstm_decoder = LSTM(9)(lstm_encoder)
output = Dense(2, activation='softmax')(lstm_decoder)
model = Model(input=main_input, output=output)
`


**Input** to the model contains Nsamples each having *9 Timesteps* wherein each timestep has *1 attribute*

We pass this input **Nx9x1** to a **LSTM ENCODER** with output dimension as 200.

Since the next node in this Neural Network is also LSTM [we obviously need a 3-dim Attribute] So Either `return_sequences=True` or use `RepeatVector(3)` layer

This output **Activation** from 1st LSTM is passed on 2nd LSTM ie **LSTM DECODER** with output dimension 9.

This output from LSTM Decoder is further passed on to a **Dense Layer with Softmax** inorder to get probability distribution of the classifier.



>**Note**:

 `model.add(LSTM(output_dim1, input_shape=(seq_len, input_dim)))`

  >Output= num_samples x output_dim1

 `model.add(LSTM(output_dim2, return_sequences=True))`

  >Output = num_samples x seq_len x output_dim2


This apporach is similiar to normal Stacked LSTM

### Type 2



`main_input = Input(shape=(9,1), dtype='float32', name='main_input')
lstm_encoder = LSTM(18,return_sequences=True)(main_input)
interim_output = Dense(1,)(lstm_encoder)
encoder_model = Model(input=main_input, output=interim_output)`

`lstm_decoder = LSTM(encoder_model)(interim_output)
final_output = Dense(2, activation='softmax')(lstm_decoder)
decoder_model = Model(input=lstm_decoder, output=interim_output)`


**Input** to the model contains Nsamples each having *9 Timesteps* wherein each timestep has *1 attribute*

We pass this input **Nx9x1** to a **LSTM ENCODER** with output dimension as 18.

We the output of **LSTM ENCODER**  and get an **intermediate output**

**NOTE: Now we compile this encoder model alone and keep the weights activations etc.**

The output from this model after compiling is given to the **LSTM DECODER** model from which we get the **final output** from the **intermediate output**

**NOTE: Backprop doesnt happen to-and-fro b/w the encoder and decoder**

