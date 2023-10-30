from src.trainer import RNNTrainer
from src.utils import setSeeds

if __name__ == '__main__':
    setSeeds(91)
    parser = RNNTrainer.get_parser()
    trainer = RNNTrainer(parser.parse_args())
    trainer.run()