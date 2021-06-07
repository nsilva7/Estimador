import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import mysql.connector as msql
from mysql.connector import Error
try:
    conn = msql.connect(host='localhost', user='root',
                        password='123456')#give ur username, password
    if conn.is_connected():
        cursor = conn.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS Simulador_dataset")
        print("Database is created")
        cursor.execute("USE Simulador_dataset")
        cursor.execute("CREATE TABLE IF NOT EXISTS data( time int(10) DEFAULT NULL,entropy decimal(10,6) DEFAULT NULL,path_consecutiveness decimal(10,6) DEFAULT NULL,bfr decimal(10,6) DEFAULT NULL,msi decimal(10,6) DEFAULT NULL,slots int(11) DEFAULT NULL,blocked varchar(20) DEFAULT NULL,sumSlots int(11) DEFAULT NULL,sumBlockedSlots int(11) DEFAULT NULL,ratio decimal(10,6) DEFAULT NULL)")
        print("Table is created")

        directory = r'C:\Users\nicoa\OneDrive\Documentos\simulador'
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                # importacion de dataset
                print("se importa: " + os.path.join(directory, filename))
                dataset = pd.read_csv(os.path.join(directory, filename), index_col=False, delimiter=',')
                dataset.head()

                for i, row in dataset.iterrows():
                    # here %S means string values
                    sql = "INSERT INTO Simulador_dataset.data VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    cursor.execute(sql, tuple(row))
                    print("Record inserted")
                    # the connection is not auto committed by default, so we must commit to save our changes
                    conn.commit()
            else:
                continue

except Error as e:
    print("Error while connecting to MySQL", e)

