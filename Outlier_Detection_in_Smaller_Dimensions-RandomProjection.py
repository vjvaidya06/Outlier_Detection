import pandas as pd
import numpy as np
import scipy.io as sio
import lof
from pyod.models.abod import ABOD
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import streamlit as st
import math
import json
from numpy.random import default_rng
version = "RandomProjection"

def getTopOutliers(numOutliers, scores):
    scores1 = scores.tolist()
    arr = []
    for i in range(numOutliers):
        arr.append([scores1.index(max(scores1)), max(scores1)])
        scores1[scores1.index(max(scores1))] = 0
    return arr

def getScoreArray(data_input, method):
    arr = np.zeros(len(data_input))
    for i in range(10):
        loading.progress(i * 5)
        randarr = default_rng().random((len(data_input.columns),2))
        red = data_input.dot(randarr)

        if (method == "Isolation Forest (IF)" or method == "Angle Based Outlier Detection (ABOD)"):
            if (method == "Isolation Forest (IF)"):
                model = IsolationForest(n_estimators=100, max_samples= 'auto', contamination= 'auto', random_state=3)
            elif (method == "Angle Based Outlier Detection (ABOD)"):
                model = ABOD(contamination=0.1, n_neighbors=10, method='fast')
            model.fit(red.astype(float))
            outliers = getTopOutliers(len(data_input), np.abs(model.decision_function(red)))
        else:
            outliers = lof.LOF_algorithm(red, p=len(data_input), distance_metric='euclidean')

        #outliers = lof.LOF_algorithm(red, p=len(data_input), distance_metric='euclidean')
        for j in range(len(outliers)):
            arr[outliers[j][0]] = arr[outliers[j][0]] + outliers[j][1]
    return arr.tolist()


def calcOutliers(trueOutliers, maxoutliers, ScoArr, method):
    outliers = []
    for i in range(0, len(st.session_state["Outliers"])):
        outliers.append(st.session_state["Outliers"][i][0])
    outliers2 = []
    for i in range(1, maxoutliers + 1):
        outliers2.append(ScoArr.index(max(ScoArr)))
        ScoArr[ScoArr.index(max(ScoArr))] = 0
    loading.progress(75)
    for i in range(1, maxoutliers + 1):
        tcounter = 0
        counter = 0
        for j in range(i):
            if outliers2[j] in outliers:
                counter += 1
            if outliers2[j] in trueOutliers:
                tcounter += 1
        st.session_state["olSN"].append((counter/i) * 100)
        st.session_state["TSN"].append((tcounter/i) * 100)
    loading.progress(100)


data = st.file_uploader(".mat Dataset or .json data", type=['.mat', ".json"], accept_multiple_files=False, key=None, help="Input a .mat Dataset that you want to find outliers in or already computed results in the form of a json file", on_change=None, args=None, kwargs=None, disabled=False)
method = st.radio(
    "What Outlier Detection Method to use?",
    ('Local Outlier Factor (LOF)', 'Isolation Forest (IF)', 'Angle Based Outlier Detection (ABOD)'))
if data is not None and (data.name).split(".")[-1] == "mat":
    loading = st.progress(0)
    datainput = sio.loadmat(data)
    maxoutliers = 0
    trueoutliers = []

    # calculate the number of true outliers and record the index of the outliers
    for k in range(len(datainput["y"])):
        maxoutliers += datainput["y"][k]
        if (datainput["y"][k] == 1):
            trueoutliers.append(k)
    data_input = pd.DataFrame(datainput['X'])
    #print(len(data_input))
    #getScoreArray(data_input)
    if ("data" in st.session_state):
        if (data.name is not st.session_state["data"]):
            for key in st.session_state.keys():
                del st.session_state[key]
            #st.session_state["olORN"] = range(1, (int(maxoutliers[0]) + 1))
            st.session_state["olORN"] = []
            st.session_state["olORN"].extend(range(1, (int(maxoutliers[0]) + 1)))
            
            #percentage retained for every number of outliers
            st.session_state["olSN"] = []
            st.session_state["TSN"] = []
            if (method == "Isolation Forest (IF)" or method == "Angle Based Outlier Detection (ABOD)"):
                if (method == "Isolation Forest (IF)"):
                    model = IsolationForest(n_estimators=100, max_samples= 'auto', contamination= 'auto', random_state=3)
                elif (method == "Angle Based Outlier Detection (ABOD)"):
                    model = ABOD(contamination=0.1, n_neighbors=10, method='fast')
                model.fit(data_input.astype(float))
                st.session_state["Outliers"] = getTopOutliers(int(maxoutliers), np.abs(model.decision_function(data_input)))
            else:
                st.session_state["Outliers"] = lof.LOF_algorithm(data_input, p=int(maxoutliers[0]), distance_metric='euclidean')
    else:
        st.session_state["olORN"] = []
        st.session_state["olORN"].extend(range(1, (int(maxoutliers[0]) + 1)))
        st.session_state["olSN"] = []
        st.session_state["TSN"] = []
        if (method == "Isolation Forest (IF)" or method == "Angle Based Outlier Detection (ABOD)"):
            if (method == "Isolation Forest (IF)"):
                model = IsolationForest(n_estimators=100, max_samples= 'auto', contamination= 'auto', random_state=3)
            elif (method == "Angle Based Outlier Detection (ABOD)"):
                model = ABOD(contamination=0.1, n_neighbors=10, method='fast')
            model.fit(data_input.astype(float))
            st.session_state["Outliers"] = getTopOutliers(int(maxoutliers), np.abs(model.decision_function(data_input)))
        else:
            st.session_state["Outliers"] = lof.LOF_algorithm(data_input, p=int(maxoutliers[0]), distance_metric='euclidean')
    fig, ax = plt.subplots()
    if ("data" not in st.session_state or data.name is not st.session_state["data"]):
        loading.progress(0)
        ScoArr = getScoreArray(data_input, method)
        calcOutliers(trueoutliers, int(maxoutliers[0]), ScoArr, method)
        st.session_state["data"] = data.name
    plot = st.empty()
    fig.supxlabel("Number of Outliers")
    fig.supylabel("Percentage of Outliers Retained")

    ax.plot(st.session_state["olORN"], st.session_state["olSN"], '-bo', label='Outliers retained with Matrix Multiplication')
    ax.plot(st.session_state["olORN"], st.session_state["TSN"], '-gX', label='True Outliers retained with Matrix Multiplication')

    combined = json.dumps([st.session_state["olORN"], st.session_state["olSN"], st.session_state["TSN"]])
    methodstring = method[(method.find("(") + 1):method.find("(", (method.find("(") + 1))]
    st.download_button("Download Results", combined, file_name=(f"{data.name[0:-4]}_{version}_{methodstring}.json"), mime='application/json', key=None, help=None, on_click=None, args=None, kwargs=None, disabled=False)

    num = max(st.session_state["olORN"]) / 20
    newlist = [1]
    for i in range (2, 20):
        newlist.append(math.ceil(num * i))
    newlist.append(max(st.session_state["olORN"]))
    plt.xticks(newlist)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.legend()
    plot = st.pyplot(fig)

if data is not None and (data.name).split(".")[-1] == "json":
    fig, ax = plt.subplots()
    d = json.load(data)
    plot = st.empty
    fig.supxlabel("Number of Outliers")
    fig.supylabel("Percentage of Outliers Retained")
    ax.plot(d[0], d[1], '-bo', label='Outliers retained with Matrix Multiplication')
    ax.plot(d[0], d[2], '-gX', label='True Outliers retained with Matrix Multiplication')
    num = max(d[0]) / 20
    newlist = [1]
    for i in range (2, 20):
        newlist.append(math.ceil(num * i))
    newlist.append(max(d[0]))
    plt.xticks(newlist)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.legend()
    plot = st.pyplot(fig)
