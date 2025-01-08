'''
Creates a basic neuron with 3 inputs.

Associated YT NNFS tutorial: https://www.youtube.com/watch?v=Wo5dMEP_BbI
'''

inputs = [1.2, 5.1, 2.1]
print(inputs)
weights = [3.1, 2.1, 8.7]
print(weights)
bias = 3.0
print (bias)

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias
print(output)
