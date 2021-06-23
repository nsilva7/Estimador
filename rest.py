from flask import Flask, json, request
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import pandas as pd

api = Flask(__name__)

@api.route('/estimador/ratio', methods=['POST'])
def get_ratio():
  data = pd.json_normalize(request.get_json())
  print(model.to_json())
  ratio =  model.predict(data).flatten()[0]

  return str(ratio)

if __name__ == '__main__':

  model = keras.models.load_model("data/model")
  # f = open("data/model/model.json", "r")
  # model_json = f.read()
  # model = keras.models.model_from_json(model_json)
  # f.close()

  api.run()