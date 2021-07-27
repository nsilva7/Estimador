import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout

#train_dataset = pd.read_csv("data/70.csv", sep=";")
#test_dataset = pd.read_csv("data/30.csv", sep=";")

#dataset = pd.read_csv("data/bfr.csv", sep=";")
#dataset = dataset.drop(columns=["ratio"])

dataset = pd.read_csv("data/datos.csv")
data = []
print("--Calculando ratios--")
for index, row in dataset.iterrows():
    if(row["time"] <= 1000 ):
        demandasTotales = 0
        demandasBloqueadas = 0
        for i in range(10):
            demandasTotales += row["demands"]
            demandasBloqueadas += row["blocked"]

        if(demandasTotales > 0):
            ratio = demandasBloqueadas/demandasTotales
        else:
            ratio = 0
        data.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],ratio])

newDataset = pd.DataFrame(data,columns=["time","entropy","msi","bfr","pc","used","shf","blocked","ratio"])
newDataset = newDataset.sample(frac=1)
data_0 = 0
data_0_ratio = []
data_01 = 0
data_01_ratio = []
data_12 = 0
data_12_ratio = []
data_23 = 0
data_23_ratio = []
data_34 = 0
data_34_ratio = []
data_45 = 0
data_45_ratio = []
data_56 = 0
data_56_ratio = []
data_67 = 0
data_67_ratio = []
data_78 = 0
data_78_ratio = []
data_89 = 0
data_89_ratio = []
data_9 = 0
data_9_ratio = []
data_1 = 0
data_1_ratio = []

dataset = []
lim = 1000
print("--Seteando datos--")
for index, row in newDataset.iterrows():
    if (row["ratio"] == 0 and data_0 < lim):
        data_0_ratio.append(row["ratio"])
        data_0 += 1
        dataset.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);
    if(row["ratio"] > 0 and row["ratio"] < 0.1 and data_01 < lim and row["ratio"] not in data_01_ratio):
        data_01_ratio.append(row["ratio"])
        data_01 += 1
        dataset.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);
    if (row["ratio"] >= 0.1 and row["ratio"] < 0.2 and data_12 < lim and row["ratio"] not in data_12_ratio):
        data_12_ratio.append(row["ratio"])
        data_12 += 1
        dataset .append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);
    if (row["ratio"] >= 0.2 and row["ratio"] < 0.3 and data_23 < lim and row["ratio"] not in data_23_ratio):
        data_23_ratio.append(row["ratio"])
        data_23 += 1
        dataset.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);
    if (row["ratio"] >= 0.3 and row["ratio"] < 0.4 and data_34 < lim and row["ratio"] not in data_34_ratio):
        data_34_ratio.append(row["ratio"])
        data_34 += 1
        dataset.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);
    if (row["ratio"] >= 0.4 and row["ratio"] < 0.5 and data_45 < lim and row["ratio"] not in data_45_ratio):
        data_45_ratio.append(row["ratio"])
        data_45 += 1
        dataset.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);
    if (row["ratio"] >= 0.5 and row["ratio"] < 0.6 and data_56 < lim and row["ratio"] not in data_56_ratio):
        data_56_ratio.append(row["ratio"])
        data_56 += 1
        dataset.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);
    if (row["ratio"] >= 0.6 and row["ratio"] < 0.7 and data_67 < lim and row["ratio"] not in data_67_ratio):
        data_67_ratio.append(row["ratio"])
        data_67 += 1
        dataset.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);
    if (row["ratio"] >= 0.7 and row["ratio"] < 0.8 and data_78 < lim and row["ratio"] not in data_78_ratio):
        data_78_ratio.append(row["ratio"])
        data_78 += 1
        dataset.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);
    if (row["ratio"] >= 0.8 and row["ratio"] < 0.9 and data_89 < lim and row["ratio"] not in data_89_ratio):
        data_89_ratio.append(row["ratio"])
        data_89 += 1
        dataset.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);
    if (row["ratio"] >= 0.9 and row["ratio"] < 1 and data_9 < lim and row["ratio"] not in data_9_ratio):
        data_9_ratio.append(row["ratio"])
        data_9 += 1
        dataset.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);
    if (row["ratio"] == 1 and data_1 < lim and row["ratio"] not in data_1_ratio):
        data_1_ratio.append(row["ratio"])
        data_1 += 1
        dataset.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);

c = 0
c01 = 0
c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0
c6 = 0
c7 = 0
c8 = 0
c9 = 0
c11 = 0


dataset = pd.DataFrame(dataset,columns=["time","entropy","msi","bfr","pc","used","shf","blocked","ratio"])

testDataset = []
for index, row in newDataset.iterrows():
    if(row['ratio'] == 0):
        c = c + 1
    else:
        testDataset.append([int(row["time"]),row["entropy"],row["msi"], row["bfr"],row["pc"],row["used"],row["shf"],row["blocked"],row["ratio"]]);
    if (row['ratio'] > 0   and row['ratio'] < 0.1):
        c01 = c01 + 1
    if (row['ratio'] > 0.1 and row['ratio'] < 0.2):
        c1 = c1 + 1
    if (row['ratio'] > 0.2 and row['ratio'] < 0.3):
        c2 = c2 + 1
    if (row['ratio'] > 0.3 and row['ratio'] < 0.4):
        c3 = c3 + 1
    if (row['ratio'] > 0.4 and row['ratio'] < 0.5):
        c4 = c4 + 1
    if (row['ratio'] > 0.5 and row['ratio'] < 0.6):
        c5 = c5 + 1
    if (row['ratio'] > 0.6 and row['ratio'] < 0.7):
        c6 = c6 + 1
    if (row['ratio'] > 0.7 and row['ratio'] < 0.8):
        c7 = c7 + 1
    if (row['ratio'] > 0.8 and row['ratio'] < 0.9):
        c8 = c8 + 1
    if (row['ratio'] > 0.9 and row['ratio'] < 1):
        c9 = c9 + 1
    if (row['ratio'] == 1):
        c11 = c11 + 1



print("--Cantidades--")
print(c)
print(c01)
print(c1)
print(c2)
print(c3)
print(c4)
print(c5)
print(c6)
print(c7)
print(c8)
print(c9)
print(c11)

#train_dataset = dataset.sample(frac=0.7,random_state=0)
#test_dataset = dataset.drop(train_dataset.index)
testDataset = pd.DataFrame(dataset,columns=["time","entropy","msi","bfr","pc","used","shf","blocked","ratio"])

train_dataset = dataset
test_dataset = newDataset


train_dataset = train_dataset.drop(columns=[ 'time'])
test_dataset = test_dataset.drop(columns=[ 'time'])

plt.figure(figsize=(8, 4))
sns.pairplot(train_dataset[["entropy", "bfr", "shf", "used", "blocked"]], diag_kind="kde")
plt.show()

train_stats = train_dataset.describe()
train_stats.pop("ratio")
train_stats = train_stats.transpose()


train_labels = train_dataset.pop('ratio')
test_labels = test_dataset.pop('ratio')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#normed_train_data = train_dataset
#normed_test_data = test_dataset


def build_model():
    model = keras.Sequential([
        layers.Dense(8, activation='relu', input_shape=[len(train_dataset.keys())]),
        #layers.Dense(64, activation='relu'),
        #layers.Dense(64, activation='relu'),
        #layers.Dense(64, activation='relu'),
        #layers.Dense(32, activation='relu'),
        #layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print(str(epoch) + "\n", end='')


EPOCHS = 1000


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Ratio]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')
    #plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [Ratio^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error')
    #plt.ylim([0,20])
    plt.legend()
    plt.show()


model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} Ratio".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [ratio]')
plt.ylabel('Predictions [ratio]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()
#
# error = test_predictions - test_labels
# plt.hist(error, bins = 25)
# plt.xlabel("Prediction Error [Ratio]")
# _ = plt.ylabel("Count")
#
# plt.show()
stats = {"mean": {
    "entropy": train_stats["mean"]["entropy"],
    "pc": train_stats["mean"]["pc"],
    "bfr": train_stats["mean"]["bfr"],
    "shf": train_stats["mean"]["entropy"],
    "msi": train_stats["mean"]["msi"],
    "used": train_stats["mean"]["used"],
    "blocked": train_stats["mean"]["blocked"]
},
"std":{
    "entropy": train_stats["std"]["entropy"],
    "pc": train_stats["std"]["pc"],
    "bfr": train_stats["std"]["bfr"],
    "shf": train_stats["std"]["shf"],
    "msi": train_stats["std"]["msi"],
    "used": train_stats["std"]["used"],
    "blocked": train_stats["std"]["blocked"]
}}


# print(stats.__str__())
#
f = open("data/model/train_stats.json", "w")
f.write(stats.__str__())
f.close()

f = open("data/model/model.json", "w")
f.write(model.to_json())
f.close()

model.save("data/model")

