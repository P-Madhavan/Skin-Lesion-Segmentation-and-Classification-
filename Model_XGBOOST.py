import numpy as np
from xgboost import XGBClassifier
from Evaluation import evaluation


def Model_XGBOOST(train_data, train_tar, test_data, test_tar):
    Train_X = np.zeros((train_data.shape[0]))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (train_data.shape[1] * train_data.shape[2]))
        Train_X[i] = np.reshape(temp, (train_data.shape[1] * train_data.shape[2]))
    # fiting model no training data
    model = XGBClassifier(epoch=100, batch_size =256)
    model.fit(Train_X, train_tar)
    y_prediction = model.predict(test_data)
    pred = [round(value) for value in y_prediction]
    Eval = evaluation(pred, test_tar)
    return Eval, pred

