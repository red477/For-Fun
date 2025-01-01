import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#load data 
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv' # Source: Data Science Dojo on GitHub (https://github.com/datasciencedojo/datasets)
data = pd.read_csv(url)

#get rid of irrelevant data
data = data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1)

#fill in missing data (some data is missing and is marked as NA)
data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

#encode needed data into numbers
labelEncoder = LabelEncoder()
data["Sex"] = labelEncoder.fit_transform(data["Sex"])
data["Embarked"] = labelEncoder.fit_transform(data["Embarked"])

#sort data into result data and preData
preData, resultData = data.iloc[:, [col for col in range(data.shape[1]) if col != 2]], data.iloc[:, 2]

#split data into testing and trainig data
trainValues, testValues, trainTarget, testTarget = train_test_split(preData, resultData, test_size=0.2, random_state= 37 )

#initialize model
model = Sequential()

#add layers
model.add(Dense(125, input_dim = trainValues.shape[1], activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(1, activation = "sigmoid"))

#compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#create early stoping
earlyStopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

#train the model
model.fit(trainValues, trainTarget, epochs = 100, batch_size = 8, validation_split = 0.1, callbacks = [earlyStopping], verbose = 1)

#display evaluation of test data
loss, accuracy = model.evaluate(testValues, testTarget)
print("Loss:", loss)
print("Accuracy:", accuracy)
