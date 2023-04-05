import argparse
import logging
import sys

from click import prompt

from train.train_pointnet import train as pointtnet_train
from test.test_pointnet import test
from train.train_protonet import train as protonet_train

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

logger.addHandler(ch)


def run_test(config_uri: str):
    logger.info(msg='=================Testing begins====================')
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description= 'Training ModelNet40',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-c', '--config', metavar='<config>', type=str, help = "Training and test config file")

    args = parser.parse_args()
    config_uri: str = args.config

    logger.info('Please choose the option that you wish to run:')

    option: int = prompt('1. Train a dataset on PointNet\n2. Train ProtoNet for fewshot learning\n3. Test your model\nOption', value_proc=int)
    print()

    if option == 1:
        logger.info(msg='=================Supervised Training begins====================')
        pointtnet_train(config_uri)
    elif option == 2:
        logger.info(msg='=================FewShot Training begins====================')
        protonet_train(config_uri)
    elif option == 3:
        run_test(config_uri)
    else:
        raise ValueError('Provided wrong input')
