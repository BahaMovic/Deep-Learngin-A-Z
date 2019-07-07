# import libaries 

import pandas as pd

# get csv dataset 

dataset = pd.read_csv('Churn_Modelling.csv')

# set X and Y 

x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# convert labels to number

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEncoder_country = LabelEncoder()
x[:,1] = labelEncoder_country.fit_transform(x[:,1])

labelEncoder_Gender = LabelEncoder()
x[:,2] = labelEncoder_Gender.fit_transform(x[:,2])

onehotencoder = OneHotEncoder(categorical_features = [1],sparse=False)
x = onehotencoder.fit_transform(x).toarray()

# ِAvoid the dummy variable

x = x[:,1:]

# Train test DataSet

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=.2)


# import ANN model liberary keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Build Ann Model 
 
classifier = Sequential()

# create hidden layers
classifier.add(Dense(init='uniform',input_dim=9,output_dim = 6,activation='relu'))

# create Second Layers
classifier.add(Dense(init='uniform',output_dim=6,activation='relu'))

# create output Layers 
classifier.add(Dense(init='uniform',output_dim=1,activation='sigmoid'))


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(x_train,y_train,batch_size=10,epochs=10)

y_pred = classifier.predict(x_test)





















