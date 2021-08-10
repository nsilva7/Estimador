import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

datos = pd.read_csv("data/estadisticas.csv")

resultados_fijo = []
resultados_ia = []
resultados = []

for index, row in datos.iterrows():
    resultados_fijo.append([row["dt_fijo_bloqueos"],row["dt_fijo_reruteos"]])
    resultados_ia.append([row["ia_bloqueos"], row["ia_reruteos"]])
    resultados.append([row["dt_fijo_bloqueos"], row["dt_fijo_reruteos"]])
    resultados.append([row["ia_bloqueos"], row["ia_reruteos"]])

resultados_fijo = np.array(resultados_fijo)
resultados_ia = np.array(resultados_ia)
resultados = np.array(resultados)


def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] <=scores[i]) and any(scores[j] < scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


pareto = identify_pareto(resultados)
print ('Pareto front index vales')
print ('Points on Pareto front: \n',pareto)

pareto_front = resultados[pareto]
print ('\nPareto front scores')
print (pareto_front)

pareto_front_df = pd.DataFrame(pareto_front)
pareto_front_df.sort_values(0, inplace=True)
pareto_front = pareto_front_df.values


x_fijo = resultados_fijo[:, 1]
y_fijo = resultados_fijo[:, 0]

plt.scatter(x_fijo, y_fijo,color="r")

x_ia = resultados_ia[:, 1]
y_ia = resultados_ia[:, 0]

x_pareto = pareto_front[:, 1]
y_pareto = pareto_front[:, 0]

plt.plot(x_pareto, y_pareto, color='b')

plt.scatter(x_ia, y_ia, color="b")
plt.xlabel('Reconfiguraciones')
plt.ylabel('Bloqueos')
plt.show()


# sin_desfragmentar_bloqueos = 0
# dt_fijo_bloqueos = 0
# ia_bloqueos = 0
# dt_fijo_reconf = 0
# ia_reconf = 0
# count = 0
# for index, row in datos.iterrows():
#     sin_desfragmentar_bloqueos += row["sin_desfragmentar_bloqueos"]
#     dt_fijo_bloqueos += row["dt_fijo_bloqueos"]
#     ia_bloqueos += row["ia_bloqueos"]
#     dt_fijo_reconf += row["dt_fijo_reruteos"]
#     ia_reconf += row["ia_reruteos"]
#     count = count + 1
#
# sin_desfragmentar_bloqueos = sin_desfragmentar_bloqueos/count
# dt_fijo_bloqueos = dt_fijo_bloqueos/count
# ia_bloqueos = ia_bloqueos/count
# ia_reconf = ia_reconf/count
# dt_fijo_reconf = dt_fijo_reconf/count
#
# dt_fijo_mejora = 100 - (dt_fijo_bloqueos*100)/sin_desfragmentar_bloqueos;
# ia_mejora = 100 - (ia_bloqueos*100)/sin_desfragmentar_bloqueos;
#
# print("RESULTADOS:")
# print("BLOQUEOS SIN DEFRAGMENTAR: " + str(round(sin_desfragmentar_bloqueos)))
# print("BLOQUEOS DT FIJO: " + str(round(dt_fijo_bloqueos)) + " -------------- MEJORA: " + str(round(dt_fijo_mejora,2)) + "%")
# print("BLOQUEOS IA: " + str(round(ia_bloqueos)) + " -------------- MEJORA: " + str(round(ia_mejora,2)) + "%")
# print("RECONFIGURACIONES DT FIJO: " + str(round(dt_fijo_reconf)))
# print("RECONFIGURACIONES IA: " + str(round(ia_reconf)))
