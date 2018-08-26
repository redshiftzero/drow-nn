from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM


class DrowSeq2Seq(object):
    def __init__(self, X, y):
        self.obj = Sequential([
            LSTM(256, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            Dense(y.shape[1], activation='softmax'),
        ])
