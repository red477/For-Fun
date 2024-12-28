import pandas as pd #Handles and proccesses data in tables
import numpy as np 
from sklearn.model_selection import train_test_split #Splits data into training and testing sets
from sklearn.preprocessing import LabelEncoder, StandardScaler #Converts text labels to numbers, Scales numerical features to have a mean of 0 and a standard deviation of 1
from tensorflow.keras.models import Sequential #Helps you build a simple feedforward neural network where layers are stacked one after another.
from tensorflow.keras.layers import Dense #Creates layers of neurons where every neuron connects to all neurons in the previous layer.
from tensorflow.keras.utils import to_categorical #Converts numeric labels (like 0, 1, 2) into one-hot encoded format (like [1,0,0])

#load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepalLength", "sepalWidth" , "petalLength", "petalWidth", "species"] #corresponding to the data set
data = pd.read_csv(url, header=None, names = columns) #loads a CSV (Comma-Separated Values) file into a pandas DataFrame (a table-like structure in Python).

#encode target lables into numeric values
labelEncoder = LabelEncoder()
data["species"] = labelEncoder.fit_transform(data["species"]) #converts the species text labels into corresponding numbers

#split data into features and target
features = data.iloc[:, :-1].values #all rows and every col but the last one
target = data.iloc[:, -1].values #all row and last col

#Standardize the feature values to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
features = scaler.fit_transform(features)

#Convert target labels to on-hot encoding
target = to_categorical(target)

#split into testing and testing datasets
featuresTrain, featuresTest, targetTrain, targetTest = train_test_split(features, target, test_size= 0.2, random_state= 40) #80% of data is used for training and 20% for testing, random_state ensures the split is the same every time the code is run 

#initialize the model
model = Sequential()

#add layers to model
model.add(Dense(50, input_dim = featuresTrain.shape[1], activation = "relu" )) #Hidden layer: 50 nodes, input from featuresTrain (.shape[1] 4 inputs one from each col), introduces non-linearity using f(x) = max(0, x) 
model.add(Dense(3, activation = "softmax" )) #output layer: 3 nodes(for the 3 species it could be), converts logits (raw predictions) into probabilities that sum to 1 using f(x_i) = exp(x_i) / sum(exp(x))

#compile model (configures the model with the necessary settings for training, including specifying the optimizer, the loss function, and the evaluation metrics)
model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ["accuracy"]) #optimizer adjusts the weights of the model during training (Adam is one of the most commonly used optimizers), loss function calculates how far the model's predictions are from the actual labels(Categorical crossentropy is used for multi-class classification problems where each target label is one-hot encoded), track the performance of the model by comparing the predictions against the true values

#train model
model.fit(featuresTrain, targetTrain, epochs = 100, batch_size = 7, validation_split = 0.1, verbose = 1) #trains 100 times, 7 samples passed through at a time, 90% of data used as training data and 10% used for validation, (verbose=1) controls how much data is displayed when training

#display the evaluation of test data
loss, accuracy = model.evaluate(featuresTest, targetTest)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
