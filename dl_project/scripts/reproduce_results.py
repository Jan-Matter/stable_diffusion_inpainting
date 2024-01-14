from omegaconf import OmegaConf
from pathlib import Path
import sys
import os
import shutil

sys.path.append(str(Path(__file__).parent.parent))


from dl_project.scripts.train_direction_model import DirectionModelTrainer
from dl_project.scripts.edit_from_directional_model import DirectionModelInference

def run_experiment_1():

    #edit_from_model
    conf = OmegaConf.load('dl_project/configs/direction_model_inference.yaml')['inference']
    conf['strength'] = 0.2
    image_decoder = DirectionModelInference(conf)

    captions = ["", "Apply lipstick to lips", "Make person laught", "Close the persons eyes"]
    for caption in captions:
        image_decoder.edit(1, caption=caption)
        shutil.copy(
             "dl_project/output/direction_model_inference/image_1_orig_decoded.png",
             f"dl_project/output/experiment1/image_1_prompt_{caption}_strength_02.png")
    
    image_decoder.edit(1, caption=None)
    for direction in range(3):
        shutil.copy(
             f"dl_project/output/direction_model_inference/image_1_direction_{direction}.png",
             f"dl_project/output/experiment1/image_1_direction_{direction}_strength_02.png")
    
    shutil.copy(
            f"dl_project/output/direction_model_inference/image_1_original.png",
            f"dl_project/output/experiment1/image_1_original.png")



def run_experiment_2():

    #edit_from_model
    conf = OmegaConf.load('dl_project/configs/direction_model_inference.yaml')['inference']
    conf['strength'] = 0.4
    image_decoder = DirectionModelInference(conf)

    captions = ["", "Apply lipstick to lips", "Make person laught", "Close the persons eyes"]
    for caption in captions:
        image_decoder.edit(1, caption=caption)
        shutil.copy(
             "dl_project/output/direction_model_inference/image_1_orig_decoded.png",
             f"dl_project/output/experiment2/image_1_prompt_{caption}_strength_04.png")
    
    image_decoder.edit(1, caption=None)
    for direction in range(3):
        shutil.copy(
             f"dl_project/output/direction_model_inference/image_1_direction_{direction}.png",
             f"dl_project/output/experiment2/image_1_direction_{direction}_strength_04.png")
    
    image_decoder.edit(0, caption=None)
    for direction in range(3):
        shutil.copy(
             f"dl_project/output/direction_model_inference/image_0_direction_{direction}.png",
             f"dl_project/output/experiment2/image_0_direction_{direction}_strength_04.png")
    
    captions = [""]
    for caption in captions:
        image_decoder.edit(1, caption=caption)
        shutil.copy(
             "dl_project/output/direction_model_inference/image_0_orig_decoded.png",
             f"dl_project/output/experiment2/image_0_prompt_{caption}_strength_04.png")
    
    shutil.copy(
            f"dl_project/output/direction_model_inference/image_1_original.png",
            f"dl_project/output/experiment2/image_1_original.png")



def run_experiment_3():
    pass
        
        


def main(training=True):
    #train_model
    if training:
        trainer = DirectionModelTrainer(OmegaConf.load('dl_project/configs/direction_model_training.yaml')['training'])
        trainer.train()

    run_experiment_1()
    run_experiment_2()


if __name__ == "__main__":
    main(training=False)

