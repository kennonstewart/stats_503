# modules.py
import numpy as np


def hard_classification(threshold: int, probabilities:np.ndarray) -> np.array:
    results = [0] * len(probabilities)
    for ind in range(len(probabilities)):
        if probabilities[ind][1] >= threshold:
            results[ind] = 1
    return results

def confusion_matrix(y_true:np.array, y_predicted:np.array) -> np.ndarray:
    FP = TP = FN = TN = 0 # initialize the rate values to 0
    y_true = [1 if prediction == "Yes" else 0 for prediction in y_true]
    for prediction, truth in zip(y_predicted, y_true):
        if (prediction == 1) and (truth == 1): # true positive
            TP += 1
        elif (prediction == 1) and (truth == 0): # false positive
            FP += 1
        elif (prediction == 0) and (truth == 0): # true negative
            TN += 1
        else: # false negative
            FN += 1
    conf_matrix = np.array([[TN, FN], [FP, TP]])

    return conf_matrix

def rate_calculations(test_set:np.ndarray, classifications:np.array) -> tuple:
    # calculate the confusion matrix
    matrix = confusion_matrix(test_set, classifications)

    # assign the TP, FP, TN, FN rates
    TN = int(matrix[0][0])
    FN = int(matrix[0][1])
    FP = int(matrix[1][0])
    TP = int(matrix[1][1])

    # status check
    # print(f"TP: {TP}")
    # print(f"FP: {FP}")
    # print(f"TN: {TN}")
    # print(f"FN: {FN}")

    # calculate TPR
    TPR = TP / max((TP + FN), 1)
    # calculate FPR
    FPR = FP / max((FP + TN), 1)
    # calculate TNR
    TNR = TN / max((TN + FP), 1)
    # calculate FNR
    FNR = FN / max((FN + TN), 1)

    return (TPR, FPR, TNR, FNR)

def calculate_nll(probabilities: np.array) -> float:
    sum = 0
    for no_prob, yes_prob in probabilities:
        sum -= np.log(max(yes_prob, no_prob))
    return(float(sum))

def simulate_thresholds(prediction_probabilities: np.ndarray, y_test: np.array) -> list:   
    # initialize the range of thresholds
    thresholds = np.linspace(0, 1, 101)
    results = list()
    for threshold in thresholds:
        # compute the classifications
        classifications = hard_classification(threshold, prediction_probabilities)

        # calculate the prediction rate results
        TPR, FPR, TNR, FNR = rate_calculations(y_test, classifications)

        # add to the results
        results.append((TPR, FPR, TNR, FNR))
        
    # return the results
    return (results)