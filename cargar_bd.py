import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import mysql.connector as msql
from mysql.connector import Error
try:
    conn = msql.connect(host='localhost', user='root',
                        password='admin')#give ur username, password
    if conn.is_connected():
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS simulador_dataset")
        print("Database is created")
        cursor.execute("USE simulador_dataset")
        cursor.execute("CREATE TABLE IF NOT EXISTS dataset4( "
                       "entropy decimal(10,6) DEFAULT NULL,"
                       "pc decimal(10,6) DEFAULT NULL,"
                       "bfr decimal(10,6) DEFAULT NULL,"
                       "shf decimal(10,6) DEFAULT  NULL,"
                       "msi decimal(10,6) DEFAULT NULL,"
                       "used decimal(10,6) DEFAULT NULL,"
                       "blocked tinyint(1)  DEFAULT NULL,"
                       "ratio decimal(10,6) DEFAULT NULL)"
                       )
        print("Table is created")

        directory = r'C:\Users\USUARIO\Documents\jose\documentos tesis\data'
        c = 93035
        for filename in os.listdir(directory):
            print(filename)
            if filename.endswith(".csv"):
                # importacion de dataset
                print("Se importa: " + os.path.join(directory, filename))
                dataset = pd.read_csv(os.path.join(directory, filename), index_col=False, delimiter=',')
                dataset.head()
                for i, row in dataset.iterrows():
                    # here %S means string values
                    sql = "INSERT INTO Simulador_dataset.dataset4 VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
                    cursor.execute(sql, tuple(row))
                    print(str(c) + " - Record inserted")
                    # the connection is not auto committed by default, so we must commit to save our changes
                    conn.commit()
                    c = c - 1
            else:
                continue

except Error as e:
    print("Error while connecting to MySQL", e)

