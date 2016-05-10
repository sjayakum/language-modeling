

## Variable Length Input

### STEP-1  Padding

`keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32')`

*Input: list_of_numbers [sequences]*

*Output: sequences x num_timesteps*


Here the `num_timesteps` is got by either `maxlen` value or automatically finds the largest sequence
If that sequence is shorter than `num_timesteps` they are padded with 0s

**Problem** 

Now assume that the sequences have different lengths. We should pad both input and desired sequences with zeros. But how will the objective function handle the padded values?

### STEP-2 Masking

**Masking**

Recurrent layer supports masking for input data with a variable number of timesteps [different seq_len for each sample].
To introduce masks to your data, use an Embedding layer with the `mask_zero` parameter set to `True`.

**Embedding Layer**

`model.add(Embedding(input_dim, output_dim, input_length=padding_size,mask_zero=True))`

`input_dim` is Size of the vocabulary [the maxmium integer that is occouring in the vocabulary] This is used for embedding [similar to word2vec]

`output_dim` is dimension of the dense embedding

`mask_zero =True` will take care about the padding

`input_length` is the padding size



