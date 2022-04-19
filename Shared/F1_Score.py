import numpy as np


def get_binary_F1_score(y, y_hat):
    TP = np.sum((y == 1) * (y_hat == 1))
    TN = np.sum((y == 0) * (y_hat == 0))
    FP = np.sum((y == 0) * (y_hat == 1))
    FN = np.sum((y == 1) * (y_hat == 0))

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall)

    return Accuracy, Precision, Recall, F1


def get_multi_F1_score(y, y_hat):
    num_class = np.max(y)
    TP = np.zeros(num_class)
    FP = np.zeros(num_class)
    FN = np.zeros(num_class)
    Precision = np.zeros(num_class)
    Recall = np.zeros(num_class)

    for n_cls in range(num_class):
        TP[n_cls] = np.sum((y == n_cls) * (y_hat == n_cls))
        FP[n_cls] = np.sum((y == n_cls) * (y_hat != n_cls))
        FN[n_cls] = np.sum((y != n_cls) * (y_hat == n_cls))

        if TP[n_cls] + FP[n_cls] != 0:
            Precision[n_cls] = TP[n_cls] / (TP[n_cls] + FP[n_cls])
        if TP[n_cls] + FN[n_cls] != 0:
            Recall[n_cls] = TP[n_cls] / (TP[n_cls] + FN[n_cls])

    Accuracy = np.sum(TP) / y.shape[0]
    Precision_macro = np.average(Precision)
    Recall_macro = np.average(Recall)
    F1_macro = (2 * Precision_macro * Recall_macro) / (Precision_macro + Recall_macro)

    return Accuracy, Precision_macro, Recall_macro, F1_macro


