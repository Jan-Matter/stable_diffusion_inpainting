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
        image_decoder.edit(1, caption_input=caption)
        shutil.copy(
             "dl_project/output/direction_model_inference/image_1_orig_decoded.png",
             f"dl_project/output/experiment1/image_1_prompt_{caption}_strength_02.png")
    
    image_decoder.edit(1, caption_input=None)
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
    conf['strength'] = 0.5
    image_decoder = DirectionModelInference(conf)

    captions = ["", "Apply lipstick to lips", "Make person laught", "Close the persons eyes"]
    for caption in captions:
        image_decoder.edit(1, caption_input=caption)
        shutil.copy(
             "dl_project/output/direction_model_inference/image_1_orig_decoded.png",
             f"dl_project/output/experiment2/image_1_prompt_{caption}_strength_05.png")
    
    image_decoder.edit(1, caption_input=None)
    for direction in range(3):
        shutil.copy(
             f"dl_project/output/direction_model_inference/image_1_direction_{direction}.png",
             f"dl_project/output/experiment2/image_1_direction_{direction}_strength_05.png")
    
    image_decoder.edit(0, caption_input=None)
    for direction in range(3):
        shutil.copy(
             f"dl_project/output/direction_model_inference/image_0_direction_{direction}.png",
             f"dl_project/output/experiment2/image_0_direction_{direction}_strength_05.png")
    
    captions = [""]
    for caption in captions:
        image_decoder.edit(1, caption_input=caption)
        shutil.copy(
             "dl_project/output/direction_model_inference/image_0_orig_decoded.png",
             f"dl_project/output/experiment2/image_0_prompt_{caption}_strength_05.png")
    
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
            image_decoder.edit(1, caption_input=caption)
            shutil.copy(
                 "dl_project/output/direction_model_inference/image_1_orig_decoded.png",
                 f"dl_project/output/experiment3/image_1_prompt_{caption}_alpha_{alpha}_scale_{scale}.png")


def run_experiment_4(training=True):
    #train and plot maskara dataset
    if training:
        #train_model
        training_confs = OmegaConf.load('dl_project/configs/direction_model_training.yaml')['training']
        training_confs['strength'] = 0.2
        training_confs['scale'] = 25.0
        training_confs['train_size'] = 1.0
        training_confs['epochs'] = 100
        training_confs['val_loss_path'] = 'dl_project/output/val_loss_v3_maskara.npy'
        training_confs['dataset']['name'] = 'maskara'
        training_confs['model']['direction_model']['path'] = 'dl_project/trained_models/direction_model_v3_maskara.pt'
        trainer = DirectionModelTrainer(training_confs)
        trainer.train()

    #edit_from_model
    conf = OmegaConf.load('dl_project/configs/direction_model_inference.yaml')['inference']
    conf['strength'] = 0.2
    conf['scale'] = 25.0
    conf['model']['direction_model']['path'] = 'dl_project/trained_models/direction_model_v3_maskara.pt'
    conf['dataset']['name'] = 'maskara'
    image_decoder = DirectionModelInference(conf)
    image_decoder.edit(1, caption_input=None)
    image_decoder.edit(0, caption_input=None)

    for direction in range(3):
        shutil.copy(
                f"dl_project/output/direction_model_inference/image_{direction}_original.png",
                f"dl_project/output/experiment4/image_{direction}_original_maskara.png")
        shutil.copy(
                f"dl_project/output/direction_model_inference/image_{direction}_orig_decoded.png",
                f"dl_project/output/experiment4/image_{direction}_orig_decoded_maskara.png")

        shutil.copy(
                f"dl_project/output/direction_model_inference/image_{direction}_direction_0.png",
                f"dl_project/output/experiment4/image_{direction}_direction_0_maskara.png")

        shutil.copy(
                f"dl_project/output/direction_model_inference/image_{direction}_direction_{direction}.png",
                f"dl_project/output/experiment4/image_{direction}_direction_1_maskara.png")
        shutil.copy(
                f"dl_project/output/direction_model_inference/image_{direction}_direction_2.png",
                f"dl_project/output/experiment4/image_{direction}_direction_2_maskara.png")
        
    shutil.copy(
            f"dl_project/output/val_loss_v3_maskara.npy",
            f"dl_project/output/experiment4/val_loss_v3_maskara.npy")
    
    #train and plot church edits dataset
    if training:
        #train_model
        training_confs = OmegaConf.load('dl_project/configs/direction_model_training.yaml')['training']
        training_confs['strength'] = 0.2
        training_confs['scale'] = 25.0
        training_confs['train_size'] = 1.0
        training_confs['epochs'] = 100
        training_confs['val_loss_path'] = 'dl_project/output/val_loss_v3_church_edits.npy'
        training_confs['dataset']['name'] = 'church_edits'
        training_confs['model']['direction_model']['path'] = 'dl_project/trained_models/direction_model_v3_church_edits.pt'
        trainer = DirectionModelTrainer(training_confs)
        trainer.train()

    #edit_from_model
    conf = OmegaConf.load('dl_project/configs/direction_model_inference.yaml')['inference']
    conf['strength'] = 0.2
    conf['scale'] = 25.0
    conf['model']['direction_model']['path'] = 'dl_project/trained_models/direction_model_v3_church_edits.pt'
    conf['dataset']['name'] = 'church_edits'
    image_decoder = DirectionModelInference(conf)
    image_decoder.edit(1, caption_input=None)
    image_decoder.edit(0, caption_input=None)

    for direction in range(3):
        shutil.copy(
                f"dl_project/output/direction_model_inference/image_{direction}_original.png",
                f"dl_project/output/experiment4/image_{direction}_original_church_edits.png")
        shutil.copy(
                f"dl_project/output/direction_model_inference/image_{direction}_orig_decoded.png",
                f"dl_project/output/experiment4/image_{direction}_orig_decoded_church_edits.png")

        shutil.copy(
                f"dl_project/output/direction_model_inference/image_{direction}_direction_0.png",
                f"dl_project/output/experiment4/image_{direction}_direction_0_church_edits.png")

        shutil.copy(
                f"dl_project/output/direction_model_inference/image_{direction}_direction_{direction}.png",
                f"dl_project/output/experiment4/image_{direction}_direction_1_church_edits.png")
        shutil.copy(
                f"dl_project/output/direction_model_inference/image_{direction}_direction_2.png",
                f"dl_project/output/experiment4/image_{direction}_direction_2_church_edits.png")

    shutil.copy(
            f"dl_project/output/val_loss_v3_church_edits.npy",
            f"dl_project/output/experiment4/val_loss_v3_church_edits.npy")
    



def main(training=True):
    #train_model
    if training:
        trainer = DirectionModelTrainer(OmegaConf.load('dl_project/configs/direction_model_training.yaml')['training'])
        trainer.train()
    
    if not os.path.exists("dl_project/trained_models"):
        os.makedirs("dl_project/trained_models")
    
    if not os.path.exists("dl_project/output/experiment1"):
        os.makedirs("dl_project/output/experiment1")
    if not os.path.exists("dl_project/output/experiment2"):
        os.makedirs("dl_project/output/experiment2")
    if not os.path.exists("dl_project/output/experiment3"):
        os.makedirs("dl_project/output/experiment3")
    if not os.path.exists("dl_project/output/experiment4"):
        os.makedirs("dl_project/output/experiment4")



    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4(training=training)


if __name__ == "__main__":
    main(training=True)

