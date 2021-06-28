#Helper functions to calculate the relevant parameters from the specific confusion matrices
import numpy as np


def accuracy_from_CM(CM):
    if (CM[0][0] + CM[1][1]) == 0:
        return 0
    return (CM[0][0] + CM[1][1])/np.sum(CM)


def precision_from_CM(CM):
    if CM[1][1] == 0:
        return 0
    return CM[1][1]/(CM[0][1] + CM[1][1])


def recall_from_CM(CM):
    if CM[1][1] == 0:
        return 0
    return CM[1][1]/sum(CM[1])


def F1_from_CM(CM):
    P = precision_from_CM(CM)
    R = recall_from_CM(CM)
    if P == 0 or R == 0:
        return 0
    return (2 * P * R)/(P + R)