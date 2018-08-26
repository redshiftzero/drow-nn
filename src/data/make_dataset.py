# -*- coding: utf-8 -*-
import click
import itertools
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    first_names = pd.read_csv(os.path.join(input_filepath, "first.txt"), 
                              header=None)
    surnames = pd.read_csv(os.path.join(input_filepath, "surname-and-house.txt"), 
                           header=None)

    first_names = np.reshape(first_names.values, len(first_names))
    surnames = np.reshape(surnames.values, len(surnames))

    drowlists = [first_names, surnames]

    # Now we want a big list of all name combinations
    drow_corpus = []
    for element in itertools.product(*drowlists):
        drow_corpus.append("{} {}".format(element[0], element[1]))

    pd.DataFrame(drow_corpus).to_csv(os.path.join(output_filepath, "input.txt"), 
                                     header=None, index=None)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
