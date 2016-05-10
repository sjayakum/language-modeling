

## Addition of 2 numbers
An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"

### Architecture

LSTM -> RepeatVector -> LSTM -> TDD

### Pipeline

**Variables**

1. number_samples [or mini_batch_size = 12]

2. sequence_length of each sample [or number of timesteps in some cases]

3. input_dim [or number of attributes each value at a particular timestep]

**Neural Network Structure**

- Here the sequence length is **7** because there are 7 chars in any 3-digit  addition i.e., 321+683

- We change the new_seq_len to **4** because the solution of any 3-digit addition will be a 4-digit answer i.e., 1004

- The input_dim here would a vocabulary size of **10**  [0-9] where in each digit of a 3-digit number is one-hot-encoded into one of those 10 attributes

*Input = mini_batch_size x seq_len x input_dim 3-D Tensor*

`model.add(LSTM(output_dim1, input_shape=(seq_len, input_dim)))`

*Output= num_samples x output_dim1*

-------------------------------------------------------

*Input = num_samples x output_dim1*

`model.add(RepeatVector(new_seq_len))`

*Output = num_samples x new_seq_len x output_dim1*

---------------------------------------------------------

*Input = num_samples x new_seq_len x output_dim1*

`model.add(LSTM(output_dim2, return_sequences=True))`

*Output = num_samples x new_seq_len x output_dim2*

--------------------------------------------------------------

*Input = num_samples x new_seq_len x output_dim2*

`model.add(TimeDistributedDense(final_output_dim))`

*Output = num_samples x new_seq_len x final_output_dim*

---------------------------------------------------------------

*Output = num_samples x new_seq_len x final_output_dim*

`model.add(Activation('softmax'))`

*Output = num_samples x new_seq_len x final_output_dim*

--------------------------------------------------------
But the `final_output_dim` now will be a probability distribution with class having the highest probablity indiciating the vocabulary charecter it belongs to
