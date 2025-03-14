import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from Evaluation import evaluation
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM




def Model_LSTM(train_data, train_target, test_data, test_target, sol=50):
    out, model = LSTM_train(train_data, train_target, test_data, sol)
    pred = np.asarray(out)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred


# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def LSTM_train(trainX, trainY, testX, sol):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, -1))
    testX = np.reshape(testX, (testX.shape[0], 1, -1))
    model = Sequential()
    model.add(LSTM(int(sol), input_shape=(1, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)
    # make predictions
    # trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    return testPredict, model




def Model_lstm(Data, Target, sol):
    out, model = lstm_train(Data, Target, sol[2])
    pred = np.asarray(out)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, Target)
    return Eval, pred


def lstm_train(Data, Target, sol):
    trainX = np.reshape(Data, (Data.shape[0], 1, Data.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(sol[2], input_shape=(1, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, Target, epochs=round(sol[3]), batch_size=1, verbose=2)
    # make predictions
    # trainPredict = model.predict(trainX)
    testPredict = model.predict(Data)
    return testPredict, model


def Model_LSTM(trainX, trainY, testX, test_y, sol):
    trainX = np.resize(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.resize(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(1, input_shape=(1, trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    testPredict = np.zeros((test_y.shape[0], test_y.shape[1])).astype('int')
    for i in range(trainY.shape[1]):
        model.fit(trainX, trainY[:, i], epochs = sol, batch_size=1, verbose=2)
        testPredict[:, i] = model.predict(testX).ravel()
    predict = np.round(testPredict).astype('int')
    Eval = evaluation(predict, test_y)
    return np.asarray(Eval).ravel()






