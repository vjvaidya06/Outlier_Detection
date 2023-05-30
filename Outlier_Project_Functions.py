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
