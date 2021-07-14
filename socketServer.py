import socket
import pandas as pd
from tensorflow import keras
import json
import  select


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


if __name__ == "__main__":
    model = keras.models.load_model("data/model")

    f = open("data/model/train_stats.json", "r")
    train_stats = json.load(f)
    f.close()

    HOST = 'localhost'  # Standard loopback interface address (localhost)
    PORT = 9999  # Port to listen on (non-privileged ports are > 1023)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:

                dataSocket = conn.recv(1024)
                if not dataSocket:
                    break

                print("{} message: ")
                print(dataSocket)
                jsonData = json.loads(dataSocket)
                data = pd.json_normalize(jsonData)

                normed_data = norm(data)
                ratio = model.predict(normed_data).flatten()[0]
                response = str(ratio)+"\r\n"
                print("{} response: ")
                print(response)

                conn.sendall(response.encode('utf-8'))

