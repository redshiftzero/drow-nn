from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM


SEQ_LENGTH = 100


class DrowSeq2Seq(object):
    def __init__(self, X, y):
        self.obj = Sequential([
            LSTM(85, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(85),
            Dropout(0.2),
            Dense(y.shape[1], activation='softmax'),
        ])
