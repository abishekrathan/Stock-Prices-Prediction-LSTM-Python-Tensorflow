# FE520 Final Project
# Stock price prediction using LSTM
# Abishek Lakshmirathan

import time
import warnings
import numpy as np
import pandas as pd
import urllib.request
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error


warnings.filterwarnings("ignore")

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read()
    data = f.split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print(("Compilation Time :{} ", time.time() - start))
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def csv_generator(stock):
    fileLine = stock + '.csv'
    urltovisit = 'https://www.quandl.com/api/v1/datasets/WIKI/{0}.csv?column=4&sort_order=asc&collapse=daily&trim_start=2000-01-01&trim_end=2016-12-31'.format(stock)
    with urllib.request.urlopen(urltovisit) as f:
        sourceCode = f.read().decode('utf-8')
        splitSource = sourceCode.split('\n')
        data = [x.split(',') for x in splitSource]
        df = pd.DataFrame(data[1:], columns=data[0])
        df_close = pd.to_numeric(df['Close'])
        df_close[:-1].to_csv(fileLine, index=False)
        print("Pulled {0}".format(stock))

def lstm_pred(stock):
    csv_generator(stock)
    X_train, y_train, X_test, y_test = load_data('{0}.csv'.format(stock), 50, True)
    type(Sequential()) == Sequential
    # Build Model
    model = Sequential()

    model.add(LSTM(
        input_dim=1,
        output_dim=50,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print('compilation time {}: ', time.time() - start)

    # Train the model
    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=10,
        validation_split=0.05)

    predictions = predict_point_by_point(model, X_test)
    plot_results(predictions, y_test)


    #Calculate RMS error of training and testing set
    dataframe = pd.read_csv('{0}.csv'.format(stock), usecols=[0], engine='python', skipfooter=3)
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    trainPredict = scaler.inverse_transform(train_pred)
    trainY = scaler.inverse_transform([y_train])
    testPredict = scaler.inverse_transform(test_pred)
    testY = scaler.inverse_transform([y_test])
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

lstm_pred("MSFT")