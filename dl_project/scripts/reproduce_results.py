from omegaconf import OmegaConf
from pathlib import Path
import sys
import os
import shutil

sys.path.append(str(Path(__file__).parent.parent))


from dl_project.scripts.train_direction_model import DirectionModelTrainer
from dl_project.scripts.edit_from_directional_model import DirectionModelInference

def run_experiment_1():
    """
    This experiment should show the effect of the scale parameter when being low enough on the image
    """

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
    """
    This experiment should show the effect of the scale parameter when being high enough on the image
    """

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
    """
    This experiment should show the effect of the scale parameter and the alpha parameter and how they interact.
    """

    alpha_range = [0.1, 0.5, 5, 50, 250]
    scale_range = [0.0, 5.0, 25.0, 100.0, 250.0]
    caption = "Apply lipstick to lips."

    #edit_from_model
    conf = OmegaConf.load('dl_project/configs/direction_model_inference.yaml')['inference']

    for alpha in alpha_range:
        for scale in scale_range:
            #edit_from_model
            conf = OmegaConf.load('dl_project/configs/direction_model_inference.yaml')['inference']
            conf['model']['direction_model']['alpha'] = alpha
            conf['scale'] = scale
            image_decoder = DirectionModelInference(conf)
            image_decoder.edit(1, caption=caption)
            shutil.copy(
                 "dl_project/output/direction_model_inference/image_1_orig_decoded.png",
                 f"dl_project/output/experiment3/image_1_prompt_{caption}_alpha_{alpha}_scale_{scale}.png")

def run_experiment_4():
    pass





        


def main(training=True):
    #train_model
    if training:
        trainer = DirectionModelTrainer(OmegaConf.load('dl_project/configs/direction_model_training.yaml')['training'])
        trainer.train()

    #run_experiment_1()
    #run_experiment_2()
    run_experiment_3()


if __name__ == "__main__":
    main(training=False)

