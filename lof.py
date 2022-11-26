import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from mat4py import loadmat
import numpy as np
import scipy.io as sio

#data_input = pd.read_csv('../../data/clikstream_data.csv')
#datainput = loadmat('C:/Users/vjvai/Documents/A/python_work/Seperate Projects/Outlier Project/lympho.mat')
#datainput = sio.loadmat('C:/Users/vjvai/Documents/A/python_work/Seperate Projects/Outlier Project/lympho.mat')
#data_input = pd.DataFrame(np.hstack((datainput['X'], datainput['y'])))
#data_input.head()
#print(data_input)

# Reachdist function


def reachdist(distance_df, observation, index):
    return distance_df[observation][index]

 # LOF algorithm implementation from scratch


def LOF_algorithm(data_input, distance_metric="cityblock", p=5):
    #print(data_input.values)
    distances = pdist(data_input.values, metric=distance_metric)
    dist_matrix = squareform(distances)
    distance_df = pd.DataFrame(dist_matrix)

    k = 2 if distance_metric == "cityblock" else 3
    observations = distance_df.columns
    lrd_dict = {}
    n_dist_index = {}
    reach_array_dict = {}

    for observation in observations:
        dist = distance_df[observation].nsmallest(k+1).iloc[k]
        indexes = distance_df[distance_df[observation]
                              <= dist].drop(observation).index
        n_dist_index[observation] = indexes

        reach_dist_array = []
        for index in indexes:
            # make a function reachdist(observation, index)
            dist_between_observation_and_index = reachdist(
                distance_df, observation, index)
            dist_index = distance_df[index].nsmallest(k+1).iloc[k]
            reach_dist = max(dist_index, dist_between_observation_and_index)
            #print("obsv")
            #print(dist_between_observation_and_index)
            # print("LOF")
            # print(dist_index)
            # print(dist_between_observation_and_index)
            #print(reach_dist)
            # print("LOF")
            reach_dist_array.append(reach_dist)
        lrd_observation = len(indexes)/sum(reach_dist_array)
        reach_array_dict[observation] = reach_dist_array
        lrd_dict[observation] = lrd_observation

    # Calculate LOF
    LOF_dict = {}
    for observation in observations:
        lrd_array = []
        for index in n_dist_index[observation]:
            lrd_array.append(lrd_dict[index])
        LOF = sum(
            lrd_array)*sum(reach_array_dict[observation])/np.square(len(n_dist_index[observation]))
        LOF_dict[observation] = LOF
    return sorted(LOF_dict.items(), key=lambda x: x[1], reverse=True)[:p]


#LOF_algorithm(data_input, p = 5)
# [(19, 11.07),
#  (525, 8.8672286617492091),
#  (66, 5.0267857142857144),
#  (638, 4.3347272196829723),
#  (177, 3.6292633292633294)]

#print(LOF_algorithm(data_input, p=5, distance_metric='euclidean'))
# [(638, 3.0800716645705695),
#  (525, 3.0103162562616288),
#  (19, 2.8402916620868903),
#  (66, 2.8014102661691211),
#  (65, 2.6456528412196416)]
