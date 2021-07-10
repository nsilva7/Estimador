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
df= pd.read_csv("data/70.csv",sep=",")
df_test= pd.read_csv("data/70.csv",sep=",")
#df.info()

#df = df.drop(columns=['time','blocked','slots','sumSlots','sumBlockedSlots'])
#df_test = df_test.drop(columns=['time','blocked','slots','sumSlots','sumBlockedSlots'])

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
sns.scatterplot(x='ratio',y='shf',data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='ratio',y='used',data=df)
plt.show()

plt.figure(figsize=(12,8))
sns.scatterplot(x='entropy',y='blocked',data=df,hue='ratio')
plt.show()

#Separacion de datos para entrenamiento y testing
X = df.drop(columns=['ratio'])
y = df['ratio']

# labelEncoder_blocked = LabelEncoder()
# X.iloc[:,5] = labelEncoder_blocked.fit_transform(X.iloc[:,5])


#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

X_train = X
y_train = y

X_test = df_test.drop(columns=['ratio'])
y_test = df_test['ratio'];

#Escalado

scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)

print("---------")
print(scaler.__getattribute__("min_"))
print(scaler.__getattribute__("scale_"))
print((scaler.__getattribute__("data_min_")))
print(scaler.__getattribute__("data_max_"))
print(scaler.__getattribute__("data_range_"))
print(scaler.__getattribute__("n_samples_seen_"))




X_test = scaler.transform(X_test)
#Creacion del modelo
model = Sequential()
# model.add(Dense(4,activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(8,input_dim=5,activation='relu'))
model.add(Dropout(0.5))
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

print("Prediction: " + str(predictions[20896]))
f = open("data/model/model.json", "w")
f.write(model.to_json())
f.close()

model.save("data/model")