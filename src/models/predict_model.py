import click
import logging
import sys

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_yaml
from keras.utils import np_utils

from src.models import network


@click.command()
@click.argument('network_weights', type=click.Path(exists=True))
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('model_yaml', type=click.Path(exists=True))
def main(network_weights, input_file, model_yaml):

    logger = logging.getLogger(__name__)
    logger.info('sampling from model....')

    # Load raw data into memory
    with open(input_file, "rb") as f:
        drow_corpus = f.read()
        drow_corpus = drow_corpus.lower()

    chars = sorted(list(set(drow_corpus)))
    n_chars = len(drow_corpus)
    n_vocab = len(chars)
    char_to_int_mapping = dict((char, index) for index, char in enumerate(chars))
    int_to_char_mapping = dict((index, char) for index, char in enumerate(chars))

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

    with open(model_yaml, 'r') as fyaml:
        model_definition = fyaml.read()
    model = model_from_yaml(model_definition)
    model.load_weights(network_weights)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # random seed
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]

    for i in range(200):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = chr(int_to_char_mapping[index])
        seq_in = [int_to_char_mapping[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print("\nDone!")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
