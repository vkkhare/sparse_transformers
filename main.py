from argparse import ArgumentParser

import yaml

from trainer import Trainer
from utilities.logger import NoOpLogger


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-c', '--config', help='configuration file', required=True)
    parser.add_argument('-d', "--logdir", type=str, default=None)
    parser.add_argument('-l', "--logger", action="store_true", default=False)
    parser.add_argument('-t', "--test", action="store_true", default=False)
    # Add your own arguments here
    #parser.add_argument('-i', '--input', help='input', required=True)

    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as f:
        config = yaml.load(f)

    if args.logger:
        if args.logdir is None:
            raise ValueError("logdir cannot be null if logging is enabled")
        from src.utilities.logger import TBLogger
        logger = TBLogger(args.logdir)
    else:
        logger = NoOpLogger()

    trainer = Trainer(config, logger=logger)

    if args.test:
        trainer.test()
    else:
        trainer.train()

if __name__ == '__main__':
    main()
