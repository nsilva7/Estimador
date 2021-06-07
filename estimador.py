import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error,mean_absolute_error

#se lee el dataset
df=pd.read_csv("data/simulador_dataset.csv")

#df.info()

df = df.drop(columns=['slots','sumSlots','sumBlockedSlots'])

plt.figure(figsize=(8, 4))
sns.distplot(df['ratio'])
plt.show()

df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='ratio',y='entropy',data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='ratio',y='bfr',data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='ratio',y='msi',data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='ratio',y='path_consecutiveness',data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='entropy',y='bfr',data=df,hue='ratio')
plt.show()

#Separacion de datos para entrenamiento y testing
X = df.drop(columns=['ratio'])
y = df['ratio']

labelEncoder_blocked = LabelEncoder()
X.iloc[:,5] = labelEncoder_blocked.fit_transform(X.iloc[:,5])


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#Escalado

scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Creacion del modelo
model = Sequential()
model.add(Dense(6,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

#Entrenamiento
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=400, callbacks=[early_stop])

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

#Evaluacion


predictions = model.predict(X_test)
mae = mean_absolute_error(y_test,predictions)
print("Mean Absolute Error: " + str(mae))
mse = np.sqrt(mean_squared_error(y_test,predictions))
print("Mean Squared Error: " + str(mse))

