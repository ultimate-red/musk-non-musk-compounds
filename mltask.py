#Data preprocessing
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("musk_csv.csv")
del dataset['conformation_name']
del dataset['ID']
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,167].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
x[:,0] = le.fit_transform(x[:,0])
ohe = OneHotEncoder(categorical_features=[0])
x = ohe.fit_transform(x).toarray()

#splitting dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#Making ANN
#importing Keras libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising ANN
classifier = Sequential()

#Adding input layer and first hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 268 ))

#Adding second layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu' ))

#Adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting the ANN
hist = classifier.fit(x_train,y_train,batch_size=10,epochs=20,validation_data=(x_test, y_test))


#Predicting the test results
y_pred = classifier.predict(x_test)
y_pred[y_pred>=0.5] = 1
y_pred[y_pred<0.5] = 0
y_pred.dtype = int
y_pred[y_pred>0] = 1

#making confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
pre = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1s = f1_score(y_test,y_pred)

#visualising results for training data
from keras.utils import plot_model
plt.plot(hist.history['acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train'], loc='upper_left')
plt.show()

plt.plot(hist.history['loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train'], loc='upper_left')
plt.show()

#visualising results for test data
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['test'], loc='upper_left')
plt.show()

plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['test'], loc='upper_left')
plt.show()
