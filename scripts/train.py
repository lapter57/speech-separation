import sys
sys.path.append("../lib")

from config import Config
from trainer import Trainer
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action="store",
                        dest="config", required=True,
                        help="Path to a config file")
    parser.add_argument("--chkpt", action="store",
                        dest="chkpt_file", default=None,
                        help="Path to a checkpoint file")
    args = parser.parse_args()
    config = Config(args.config)
    trainer = Trainer(config, args.chkpt_file)
    trainer.train()

