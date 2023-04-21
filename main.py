import argparse
import logging
import sys

from click import prompt

from train.train_pointnet import train as pointtnet_train
from test.test_protonet import prepare_dataset as protonet_test
from test.test_pointnet import test as pointnet_test
from train.train_protonet import train as protonet_train
from config import Config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

logger.addHandler(ch)


def run_test(config: Config):
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
    config = Config.from_file(config_uri)

    logger.info('Please choose the option that you wish to run:')

    option: int = prompt('1. Train a dataset on PointNet\n2. Train ProtoNet for fewshot learning\n3. Test your model\nOption', value_proc=int)
    print()

    if option == 1:
        logger.info(msg='=================Supervised Training begins====================')
        pointtnet_train(config)
    elif option == 2:
        logger.info(msg='=================FewShot Training begins====================')
        protonet_train(config)
    elif option == 3:
        choosen_model: int = prompt('1. ProtoNet \n2. PointNet', value_proc=int)
        if choosen_model == 1:
            protonet_test(config)
        elif choosen_model == 2:
            run_test(config)
        else:
            raise ValueError('Provided wrong input')
    else:
        raise ValueError('Provided wrong input')
