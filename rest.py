from flask import Flask, json, request
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import pandas as pd
import numpy as np

api = Flask(__name__)

@api.route('/estimador/ratio', methods=['POST'])
def get_ratio():
  #data = pd.json_normalize(request.get_json())
  #print(model.to_json())
  #ratio =  model.predict(data).flatten()[0]


  dataset = pd.read_csv("data/pruebas.csv", sep=";")
  scaler = MinMaxScaler()
  scaler.min_ = [-0.00079345, -0.00013418, 0, 0]
  scaler.scale_ = [0.01705898, 0.00285233, 1.1231412, 0.00285326]
  scaler.data_min_ = [0.046512, 0.047042, 0, 0]
  scaler.data_max_ = [58.666667,350.637913,0.89036,350.47619]
  scaler.data_range_ = [58.620155, 350.590871, 0.89036, 350.47619]
  scaler.n_samples_seen_ = 48754
  data= scaler.transform(dataset)
  predictions = model.predict(data)
  c = 0
  for p in predictions:
    print("Prediction: " + str(p))
    c = c + 1


  return "Success"

if __name__ == '__main__':

  model = keras.models.load_model("data/model")
  # f = open("data/model/model.json", "r")
  # model_json = f.read()
  # model = keras.models.model_from_json(model_json)
  # f.close()

  api.run()

