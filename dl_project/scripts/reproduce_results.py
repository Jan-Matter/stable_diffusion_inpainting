from omegaconf import OmegaConf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


from dl_project.scripts.train_direction_model import DirectionModelTrainer
from dl_project.scripts.edit_from_directional_model import DirectionModelInference


def main():
    #train_model
    trainer = DirectionModelTrainer(OmegaConf.load('dl_project/configs/direction_model_training.yaml')['training'])
    trainer.train()

    #edit_from_model
    image_decoder = DirectionModelInference(OmegaConf.load('dl_project/configs/direction_model_inference.yaml')['inference'])
    image_decoder.edit(0)
    image_decoder.edit(1)
    image_decoder.edit(2)

