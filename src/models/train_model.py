import click
import datetime
import logging

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from src.models import network

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path(exists=True))
@click.option('--num_epochs', default=20)
def main(input_file, output_filepath, num_epochs):

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
    model = network.DrowSeq2Seq(X, y)
    model.obj.compile(loss='categorical_crossentropy', optimizer='adam')
    filepath = 'models/drow-weights-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                                 save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.obj.fit(X, y, epochs=num_epochs, batch_size=128, callbacks=callbacks_list)

    # serialize model to YAML
    model_yaml = model.obj.to_yaml()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
    with open("models/model-config.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.obj.save_weights("models/model-{}.hdf5".format(timestamp))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
