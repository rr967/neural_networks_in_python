Definitions and principles for neural networks
==============================================

inputs = values you want to track, 
          features from a single status,
  **caution**: you can never change the inputs directly, 
            because they are or real (input) data or outputs values from a previous hidden layer
neuron

weights = are used to influence the result of the (next) calculation (magnitude)

bias = are used to influence the result of the (next) calculation (offset)

batches calculate in parallel helps in generalisation

batch size = max 32 otherwise overfitting

loss

optimizer (changes weights and biases to set them to the optimal value)

activation point / function output = ReLu ( input * weight + bias )

back propagation

learning rate = how much of previous knowledge do we want to keep vs new knowledge

epoch = iteration

When using numpy library ( for mathematical operations on series/list of numbers ):

list = 1D array ( vector )

list of lists ( lol ) = 2D array ( matrix )

list of lists of lists ( lolol ) = 3D array ( tensor )
