import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
def compare(A, B, Origlen):
    listnums = [0, 0, 0, 0]
    #listnums[2] is how many are the same between them
    for i in range(len(A)):
        #if A[i][0] in B:
        for j in range(len(B)):
            if A[i][0] == B[j][0]:
                listnums[2] += 1
    listnums[0] = len(A) - listnums[2]
    listnums[1] = len(B) - listnums[2]
    listnums[3] = Origlen - (len(A) + len(B) - listnums[2])
    return listnums

def simpleplot(listx, listy, labelx, labely, wholenumsx, wholenumsy, type, **minmax):
    if type != NULL:
        plt.plot(listy, listx, type)
    else:
        plt.plot(listy, listx, type)
    plt.ylabel(labely)
    plt.xlabel(labelx)
    if wholenumsx == True:
        new_list = range(math.floor(min(listy)), math.ceil(max(listy))+1)
        plt.xticks(new_list)
    if wholenumsy == True:
        new_list = range(math.floor(min(listx)), math.ceil(max(listx))+1)
        plt.yticks(new_list)
    try:
        plt.xlim(minmax["xmin"], minmax["xmax"])
    except KeyError:
        NULL
    try:
        plt.ylim(minmax["ymin"], minmax["ymax"])
    except KeyError:
        NULL
    plt.show()
