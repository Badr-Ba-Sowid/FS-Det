import argparse
import logging
import sys

from click import prompt

from train.train_pointnet import train
from test.test_pointnet import test

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

logger.addHandler(ch)

def run_train(config_uri: str):
    logger.info(msg='=================Training begins====================')

    train(config_uri)

def run_test(config_uri: str):
    logger.info(msg='=================Testing begins====================')
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description= 'Training Using PointNet Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', metavar='<config>', type=str, help = "Training and test config file")

    args = parser.parse_args()
    config_uri: str = args.config

    logger.info('Please choose the option that you wish to run:')

    option: int = prompt('1. Train a dataset on PointNet\n2. Test your model\nOption', value_proc=int)
    print()

    if option == 1:
        run_train(config_uri)
    elif option == 2:
        run_test(config_uri)
    else:
        raise ValueError('Provided wrong input')
