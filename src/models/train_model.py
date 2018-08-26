import click
import logging

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils


@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(exists=True))
def main(input_file, output_filepath):

    logger = logging.getLogger(__name__)
    logger.info('beginning training...')

    # Load training data into memory
    with open(input_file, "r") as f:
        drow_corpus = f.read()
        drow_corpus = drow_corpus.lower()

    chars = sorted(list(set(drow_corpus)))
    char_to_int_mapping = dict((char, index) for index, char in enumerate(chars))

    n_chars = len(drow_corpus)
    n_vocab = len(chars)
    logger.info('{} total chars in training corpus'.format(n_chars))
    logger.info('{} unique chars in training corpus'.format(n_vocab))

    # Generate sequences for training
    seq_length = 6
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = drow_corpus[i:i + seq_length]
        seq_out = drow_corpus[i + seq_length]
        dataX.append([char_to_int_mapping[char] for char in seq_in])
        dataY.append(char_to_int_mapping[seq_out])
    n_patterns = len(dataX)
    logger.info('{} total patterns for training'.format(n_patterns))

    # Reshape to [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))

    # Normalize
    X = X / float(n_vocab)

    # One hot encode the output variable
    y = np_utils.to_categorical(dataY)

    # LSTM RNN model
    model = Sequential([
        LSTM(256, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        Dense(y.shape[1], activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    filepath = 'drow-weights-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
