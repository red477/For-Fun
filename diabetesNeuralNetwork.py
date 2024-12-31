import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/refs/heads/master/diabetes.csv"

# Pima Indians Diabetes Dataset:
# Pradaschnor, N. (n.d.). Pima Indians Diabetes Dataset.
# Retrieved from https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/refs/heads/master/diabetes.csv

data = pd.read_csv(url) #just read it - already has labels

#split data into features and target
features = data.iloc[:, :-1].values
result = data.iloc[:, -1].values

#standardize the features
scaler = StandardScaler()
features = scaler.fit_transform(features)

#dont need on-hot encoding for binary
#result = to_categorical(result)

#split into testing and target datasets
featuresTrain, featuresTest, resultTrain, resultTest = train_test_split(features, result, test_size = 0.2, random_state=40)

#initialize the model
model = Sequential()

#create layers
model.add(Dense(128, input_dim=featuresTrain.shape[1], activation="relu"))
model.add(Dropout(0.3))  #dropout to 30%
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))  #dropout to 30%
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.3))  #dropout to 30%
model.add(Dense(1, activation="sigmoid"))

#compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#early stoping
"""
Initializes an EarlyStopping callback to monitor the validation loss ('val_loss') during training.
If the validation loss does not improve for 10 consecutive epochs (specified by 'patience=10'),
training will stop early to prevent overfitting or wasting resources on further training.
Additionally, the 'restore_best_weights=True' parameter ensures that after training stops,
the model's weights are reverted to the state where the validation loss was at its minimum,
providing the best-performing version of the model on the validation data.
"""
earlyStopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

#training
model.fit(featuresTrain, resultTrain, epochs=100, batch_size=16, validation_split=0.1, callbacks=[earlyStopping], verbose=1)

#display evaluation of test data
loss, accuracy = model.evaluate(featuresTest, resultTest)
print("Loss:", loss)
print("Accuracy:", accuracy)
