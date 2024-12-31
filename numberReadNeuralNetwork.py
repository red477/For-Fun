from tensorflow.keras.datasets import mnist #this is the data - images is 28 by 28 
# MNIST Dataset:
# LeCun, Y., Cortes, C., & Burges, C. J. (1998). 
# The MNIST Database of Handwritten Digits. Retrieved from http://yann.lecun.com/exdb/mnist/
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense

#load the dataset 
(valuesTrain, targetTrain), (valuesTest, targetTest) = mnist.load_data()

#normalize the data - before the pixel values range from 0-255, now 0-1
valuesTrain = valuesTrain / 255
valuesTest = valuesTest / 255

#on-hot encode target data
targetTrain = to_categorical(targetTrain)
targetTest = to_categorical(targetTest)

#initialize the model
model = Sequential([Flatten(input_shape = (28, 28))])

#add layers
model.add(Dense(128, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(32, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

#compile model
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

#train model
model.fit(valuesTrain, targetTrain, epochs = 10, batch_size = 50)

#print loss and accuracy
loss, accuracy = model.evaluate(valuesTest, targetTest)
print(f'Loss: {loss}\nAccuracy: {accuracy}')
