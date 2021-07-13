import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout

#train_dataset = pd.read_csv("data/70.csv", sep=";")
#test_dataset = pd.read_csv("data/30.csv", sep=";")

dataset = pd.read_csv("data/bfr.csv", sep=";")
dataset = dataset.drop(columns=["ratio"])

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#train_dataset = train_dataset.drop(columns=[ 'pc', 'msi'])
#test_dataset = test_dataset.drop(columns=[ 'pc', 'msi'])

plt.figure(figsize=(8, 4))
sns.pairplot(train_dataset[["entropy", "pc", "bfr", "shf", "msi", "used", "blocked"]], diag_kind="kde")
plt.show()

train_stats = train_dataset.describe()
train_stats.pop("bfr")
train_stats = train_stats.transpose()


train_labels = train_dataset.pop('bfr')
test_labels = test_dataset.pop('bfr')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#normed_train_data = train_dataset
#normed_test_data = test_dataset


def build_model():
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=[len(train_dataset.keys())]),
        #layers.Dense(16, activation='relu'),
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
plt.xlabel('True Values [Ratio]')
plt.ylabel('Predictions [Ratio]')
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
    "shf": train_stats["mean"]["shf"],
    "msi": train_stats["mean"]["msi"],
    "used": train_stats["mean"]["used"],
    "blocked": train_stats["mean"]["blocked"]
},
"std":{
    "entropy": train_stats["std"]["entropy"],
    "pc": train_stats["std"]["pc"],
    "shf": train_stats["std"]["shf"],
    "msi": train_stats["std"]["msi"],
    "used": train_stats["std"]["used"],
    "blocked": train_stats["std"]["blocked"]
}}


print(stats.__str__())

f = open("data/model/train_stats.json", "w")
f.write(stats.__str__())
f.close()

f = open("data/model/model.json", "w")
f.write(model.to_json())
f.close()

model.save("data/model")

