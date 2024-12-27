import numpy as np 

#sigmoid funtion and derivative
def nonlin(x, deriv = False):
    if deriv:
        return x * (1-x) #derivative
    return 1/(1 + np.exp(-x)) #sigmoid funtion

#digits as 3x5 binary matrices
numbers = [
    # Number 1
    [
        [0, 1, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1]
    ],
    # Number 2
    [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ],
    # Number 3
    [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ],
    # Number 4
    [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1]
    ],
    # Number 5
    [
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ],
    # Number 6
    [
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
]

#expected outcome for each digit
#one-hot encoded output for each digit
output = np.array([
    [1, 0, 0, 0, 0, 0],  # Digit 1
    [0, 1, 0, 0, 0, 0],  # Digit 2
    [0, 0, 1, 0, 0, 0],  # Digit 3
    [0, 0, 0, 1, 0, 0],  # Digit 4
    [0, 0, 0, 0, 1, 0],  # Digit 5
    [0, 0, 0, 0, 0, 1],  # Digit 6
])

# Flatten each 3x5 digit matrix into a 1D array for processing
flattenedDigits = [np.array(num).flatten() for num in numbers]

# Makes the random numbers the same every time you run the code
np.random.seed(1)

# initialize random wights
wei1 = 2 * np.random.random((15, 100)) -1 # for between input and hidden layer
wei2 = 2 * np.random.random((100, 50)) -1 # for between hidden and output layer
wei3 = 2 * np.random.random((50, 6)) -1

#training 5000 times
for epoch in range(500):
    for i, dig in enumerate(flattenedDigits):

        # Forward propagation: Passing input through the network to calculate the output
        layer0 = dig.reshape(1,-1) #layer input, reshape changes data into a single row with all its original elements, to 2D array. ie: [1,0,1] -> [[1, 0, 1]]
        layer1 = nonlin(np.dot(layer0, wei1)) #hidden layer 1
        layer2 = nonlin(np.dot(layer1, wei2)) #hidden layer 2
        layer3 = nonlin(np.dot(layer2, wei3)) #output layer
        
        #BackPropagation

        #error of output layer
        layer3Error = output[i] - layer3
        #how much to adjusts the weights for output layer(error * sigmoid derivative)
        layer3Delta = layer3Error * nonlin(layer3, deriv = True)

        #error of hidden layer 1
        layer2Error = layer3Delta.dot(wei3.T)
        #how much to adjusts the weights for hiden layer 1(error * sigmoid derivative)
        layer2Delta = layer2Error * nonlin(layer2, deriv = True)

        #error for hidden layer 2
        layer1Error = layer2Delta.dot(wei2.T)
        #how much to adjusts the weights for hidden layer 2(error * sigmoid derivative)
        layer1Delta = layer1Error * nonlin(layer1, deriv = True)

        #update weights 
        wei1 += layer0.T.dot(layer1Delta)
        wei2 += layer1.T.dot(layer2Delta)
        wei3 += layer2.T.dot(layer3Delta)

# Test the network
correct_predictions = 0
total_predictions = len(flattenedDigits)

for i, dig in enumerate(flattenedDigits):
    layer0 = dig.reshape(1, -1)
    layer1 = nonlin(np.dot(layer0, wei1))
    layer2 = nonlin(np.dot(layer1, wei2))
    output_layer = nonlin(np.dot(layer2, wei3))

    predicted_digit = np.argmax(output_layer)  # Get the index of the highest value
    if predicted_digit == i:  # Compare the predicted digit to the actual digit
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy: {accuracy:.2f}%")
