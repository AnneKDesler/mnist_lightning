# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    for i in range(5):
        data_train = np.load(input_filepath + "/corruptmnist/train_{}.npz".format(i))
        images = data_train["images"]
        mean = images.mean((1,2))[:, np.newaxis,np.newaxis]
        std = images.std((1,2))[:, np.newaxis,np.newaxis]
        images = (images - mean)/std
        images = images[:, np.newaxis,:,:]
        labels = data_train["labels"]
        np.savez(output_filepath + "train_{}.npz".format(i), images=images, labels=labels)

    data_test = np.load(input_filepath + "/corruptmnist/test.npz")
    images = data_test["images"]
    mean = images.mean((1,2))[:, np.newaxis,np.newaxis]
    std = images.std((1,2))[:, np.newaxis,np.newaxis]
    images = (images - mean)/std
    images = images[:, np.newaxis,:,:]
    labels = data_test["labels"]
    np.savez(output_filepath + "test.npz", images=images, labels=labels)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
