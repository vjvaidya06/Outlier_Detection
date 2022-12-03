import pandas as pd
import numpy as np
import scipy.io as sio
import lof
from sklearn.decomposition import PCA
import Outlier_Project_Functions as OPF
import matplotlib.pyplot as plt
import streamlit as st
import math
from scipy.spatial import distance
import json
from pyod.models.abod import ABOD
from sklearn.ensemble import IsolationForest
version = "PCA"
#recOred determines whether it's calculating for the reduced datasets with the Euclidean distance dimension (True), or just the reduced datasets (False)
def getTopOutliers(numOutliers, scores):
    scores1 = scores.tolist()
    arr = []
    for i in range(numOutliers):
        arr.append([scores1.index(max(scores1)), max(scores1)])
        scores1[scores1.index(max(scores1))] = 0
    return arr

def calcOutliers(data_input, maxoutliers, columns, recOred, method):
    #Calculating Each Outlier Preemptively
    if (method == "Isolation Forest (IF)" or method == "Angle Based Outlier Detection (ABOD)"):
        if (method == "Isolation Forest (IF)"):
            model = IsolationForest(n_estimators=100, max_samples= 'auto', contamination= 'auto', random_state=3)
        elif (method == "Angle Based Outlier Detection (ABOD)"):
            model = ABOD(contamination=0.1, n_neighbors=10, method='fast')
        model.fit(data_input.astype(float))
        outliersP = getTopOutliers(int(maxoutliers), np.abs(model.decision_function(data_input)))
    else:
        outliersP = lof.LOF_algorithm(data_input, p=int(maxoutliers), distance_metric='euclidean')
    for j in range(1, (columns + 1)):
        loading.progress(int(((j - 1) * (100/columns))) + int((100/columns)))
        pca = PCA(n_components=j)
        if (len(st.session_state["olDRN"]) < columns - 1):
            st.session_state["olDRN"].append(j)
        y = pca.fit_transform(data_input) 
        if (recOred):
            z = pca.inverse_transform(y) 
            dist = []
            for k in range(0, len(data_input)):
                data_inputa = np.array(data_input)
                dist.append(distance.euclidean(data_inputa[k], z[k]))
            y = pd.concat([pd.DataFrame(y), pd.DataFrame(dist)], axis=1)
            y.columns = range(len(y.columns))
            y.index = range(len(y.index))


        if (method == "Isolation Forest (IF)" or method == "Angle Based Outlier Detection (ABOD)"):
            model.fit(pd.DataFrame(y))
            outliers2P = getTopOutliers(int(maxoutliers), np.abs(model.decision_function(pd.DataFrame(y))))
        else:
            outliers2P = lof.LOF_algorithm(pd.DataFrame(y), p=int(maxoutliers), distance_metric='euclidean')
        for i in range(1, (int(maxoutliers) + 1)):
            outliers = outliersP[0:i]
            #Inverse transforms the dataset, calculates all the Euclidean distances, and then adds them as an extra dimension

            if (j != columns):
                outliers2 = outliers2P[0:i]
            else:
                outliers2 = outliers
            counter = 0
            for n in range(len(outliers2)):
                if outliers2[n][0] in trueoutliers:
                    counter+=1
            if (recOred == False):
                st.session_state["TSN"][i - 1].append((counter /len(outliers2)) * 100)
                answerlist = OPF.compare(outliers, outliers2, len(data_input))
                st.session_state["olSN"][i - 1].append((answerlist[2] / i) * 100)
            else:
                st.session_state["RTSN"][i - 1].append((counter /len(outliers2)) * 100)
                answerlist = OPF.compare(outliers, outliers2, len(data_input))
                st.session_state["RSN"][i - 1].append((answerlist[2] / i) * 100)
            #answerlist[2] is the number of outliers retained
        counter = 0
        for n in range(len(outliers)):
            if outliers[n][0] in trueoutliers:
                counter+=1
        #if (recOred == False):
            #st.session_state["TSN"][i - 1].append((counter /len(outliers)) * 100)
            #st.session_state["olSN"][i - 1].append(100)
        #else:
            #st.session_state["RTSN"][i - 1].append((counter /len(outliers)) * 100)
            #st.session_state["RSN"][i-1].append(100)
        if (len(st.session_state["olDRN"]) == columns - 1):
            st.session_state["olDRN"].append(len(data_input.columns))

data = st.file_uploader(".mat Dataset or .json data", type=['.mat', ".json"], accept_multiple_files=False, key=None, help="Input a .mat Dataset that you want to find outliers in or already computed results in the form of a json file", on_change=None, args=None, kwargs=None, disabled=False)
method = st.radio(
    "What Outlier Detection Method to use?",
    ('Local Outlier Factor (LOF)', 'Isolation Forest (IF)', 'Angle Based Outlier Detection (ABOD)'))
#This executes if the received file is a mat file
if data is not None and (data.name).split(".")[-1] == "mat":
    datainput = sio.loadmat(data)
    maxoutliers = 0
    trueoutliers = []

    # calculate the number of true outliers and record the index of the outliers
    for k in range(len(datainput["y"])):
        maxoutliers += datainput["y"][k]
        if (datainput["y"][k] == 1):
            trueoutliers.append(k)
    data_input = pd.DataFrame(datainput['X'])
    if ("data" in st.session_state):
        if (data.name is not st.session_state["data"]):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.session_state["olDRN"] = []
            st.session_state["olSN"] = []
            st.session_state["TSN"] = []
            st.session_state["RSN"] = []
            st.session_state["RTSN"] = []
            for m in range(int(maxoutliers[0])):
                st.session_state["olSN"].append([])
                st.session_state["TSN"].append([])
                st.session_state["RSN"].append([])
                st.session_state["RTSN"].append([])
    else:
        st.session_state["olDRN"] = []
        st.session_state["olSN"] = []
        st.session_state["TSN"] = []
        st.session_state["RSN"] = []
        st.session_state["RTSN"] = []
        for m in range(int(maxoutliers[0])):
            st.session_state["olSN"].append([])
            st.session_state["TSN"].append([])
            st.session_state["RSN"].append([])
            st.session_state["RTSN"].append([])
    fig, ax = plt.subplots()
    #calculating everything first, storing each dimreducenums in olDRN, and each similaritynums in olSN 
    st.write("loading")
    if ("data" not in st.session_state or data.name is not st.session_state["data"]):
        loading = st.progress(0)
        calcOutliers(data_input, maxoutliers[0], len(data_input.columns), False, method)
        calcOutliers(data_input, maxoutliers[0], len(data_input.columns), True, method)
        st.session_state["data"] = data.name
        loading.progress(100)
    numofoutliers = st.slider("Number of outliers", min_value=1, max_value=int(maxoutliers[0]), value=None, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    originaloutlier = numofoutliers
    plot = st.empty()
    #plot two extra lines for the reconstructed data. Use original number of reduced dimensions for x
    fig.supxlabel("Dimensions Reduced To")
    fig.supylabel("Percentage of Outliers Retained")
    ax.plot(st.session_state["olDRN"], st.session_state["olSN"][numofoutliers - 1], '-b.', label='Outliers retained with Dim reduction')
    ax.plot(st.session_state["olDRN"], st.session_state["TSN"][numofoutliers - 1], '-gx', label='True Outliers retained')
    ax.plot(st.session_state["olDRN"], st.session_state["RSN"][numofoutliers - 1], '-yo', label='Outliers retained with Dim reduction (Reconstructed)')
    ax.plot(st.session_state["olDRN"], st.session_state["RTSN"][numofoutliers - 1], '-rX', label='True Outliers retained (Reconstructed)')
    combined = json.dumps([st.session_state["olDRN"], st.session_state["olSN"], st.session_state["TSN"], st.session_state["RSN"], st.session_state["RTSN"], maxoutliers[0]])
    methodstring = method[(method.find("(") + 1):method.find("(", (method.find("(") + 1))]
    st.download_button("Download Results", combined, file_name=(f"{data.name[0:-4]}_V{version}_{methodstring}.json"), mime='application/json', key=None, help=None, on_click=None, args=None, kwargs=None, disabled=False)
    num = max(st.session_state["olDRN"]) / 20
    newlist = [1]
    for i in range (2, 20):
        newlist.append(math.ceil(num * i))
    newlist.append(max(st.session_state["olDRN"]))
    plt.xticks(newlist)
    ax.invert_xaxis()
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.legend()
    plot = st.pyplot(fig)

#This executes if the received file is a json file
if data is not None and (data.name).split(".")[-1] == "json":
    fig, ax = plt.subplots()
    d = json.load(data)
    plot = st.empty
    fig.supxlabel("Dimensions Reduced To")
    fig.supylabel("Percentage of Outliers Retained")
    numofoutliers = st.slider("Number of outliers", min_value=1, max_value=int(d[5]), value=None, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    ax.plot(d[0], d[1][numofoutliers-1], "-b.", label='Outliers retained with Dim reduction')
    ax.plot(d[0], d[2][numofoutliers-1], "-gx", label='True Outliers retained')
    ax.plot(d[0], d[3][numofoutliers-1], "-yo", label='Outliers retained with Dim reduction (Reconstructed)')
    ax.plot(d[0], d[4][numofoutliers-1], "-rX", label='True Outliers retained (Reconstructed)')
    num = max(d[0]) / 20
    newlist = [1]
    for i in range (2, 20):
        newlist.append(math.ceil(num * i))
    newlist.append(max(d[0]))
    plt.xticks(newlist)
    ax.invert_xaxis()
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.legend()
    plot = st.pyplot(fig)
