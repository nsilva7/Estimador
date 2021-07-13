from flask import Flask, json, request
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import pandas as pd
import numpy as np
import time

api = Flask(__name__)


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

@api.route('/estimador/ratio', methods=['POST'])
def get_ratio():
  data = pd.json_normalize(request.get_json())
  normed_data = norm(data)
  ratio =  model.predict(normed_data).flatten()[0]

  return ratio.__str__()

if __name__ == '__main__':

  model = keras.models.load_model("data/model")

  f = open("data/model/train_stats.json", "r")
  train_stats = json.load(f)
  f.close()

  api.run()

