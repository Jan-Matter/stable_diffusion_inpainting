import argparse, os
import torch
from  torch.nn.parallel import data_parallel
import numpy as np
import PIL
from omegaconf import OmegaConf
from PIL import Image
from imwatermark import WatermarkEncoder
from pytorch_lightning import seed_everything
import random

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler

from dl_project.direction_models.direction_model import DirectionModel
from dl_project.loss.contrastive_loss import ContrastiveLoss
from dl_project.datasets.image_caption_dataset import ImageCaptionDataset

class DirectionModelTrainer:
     
    def __init__(self, configs):
        self.configs = configs
        self.model_save_path = configs['model']['direction_model']['path']
        self.direction_count = configs['model']['direction_model']['direction_count']
        self.direction_model = DirectionModel(**configs['model']['direction_model'])
        #self.ldm_model = self.__load_model_from_config(configs['model']['ldm_model'])
        #self.ldm_sampler = PLMSSampler(self.ldm_model)
        self.loss = ContrastiveLoss(**configs['loss'])
        self.optimizer = torch.optim.Adam(self.direction_model.parameters(), lr=configs['lr'])
        self.device = torch.device(configs['device'])
        self.direction_model.to(self.device)

        dataset = ImageCaptionDataset(configs['dataset'], training=False) #TODO change to to true when dataset is ready
        train_size = int(configs['train_size'] * len(dataset))
        self.batch_size = configs['batch_size']
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.epochs = configs['epochs']

        self.best_loss = None

        #defaults: strength = 0.8 and ddim_steps = 50 and scale = 5.0
        self.t_enc = int(configs['strength'] * configs['ddim_steps'])
        self.scale = configs['scale']
        #self.uc = self.ldm_model.get_learned_conditioning(self.batch_size * [""])

        seed_everything(configs['seed'])


    
    def train(self):
        for epoch in range(self.epochs):
            self.direction_model.train()
            for batch_idx, x in enumerate(self.train_loader):

                loss = self.__get_batch_loss(x)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
            
            val_loss = self.validate()
            print(f'Epoch {epoch}, Validation Loss {val_loss}, best Validation Loss {self.best_loss}')
            if self.best_loss is None or val_loss < self.best_loss:
                self.best_loss = val_loss
                print(f"Saving model with validation loss {val_loss}")
                torch.save(self.model.state_dict(), self.model_save_path)
    
    def __get_batch_loss(self, x):
        images = x['image'].to(self.device)
        captions = x['caption']

        """
        captions_enc = self.ldm_model.get_learned_conditioning(captions)
        noised_images_enc = self.ldm_model.get_first_stage_encoding(
            self.ldm_model.encode_first_stage(images)
        )

        # [batch_size, noised_image_enc_length] -> [batch_size, direction_count, noised_image_enc_length]
        noised_images_enc = noised_images_enc.repeat(1, self.direction_count, 1)
        
        # [batch_size, caption_enc_length] -> [batch_size, direction_count, caption_enc_length]
        directions = self.direction_model(captions_enc)

        #apply stable diffusion model to input images
        features = data_parallel(self.__decode,
                (noised_images_enc, directions), dim=-1)
        """
        features = torch.randn((self.batch_size * self.direction_count, 4, 16, 16)).to(self.device)
        feature_length = features.shape[-1]
        # reshape features to [batch_size * direction_count, feature_length].
        reshaped_features = features.reshape(
            features.shape[0] * features.shape[1], -1)
            
        # 4. create group_indices of shape[batch_size * direction_count]: which feature is from which direction model.
        group_indices = torch.stack([torch.ones(features.shape[0]) * i
                                        for i in range(features.shape[1])])\
                                        .T\
                                        .reshape(features.shape[0] * features.shape[1])

        # Optional: check if the reshaping is correct
        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                assert (reshaped_features[i * features.shape[1] + j] == features[i][j]).all()

        loss = self.loss(reshaped_features, group_indices)
        return loss

    def __decode(self, z, c):
        with torch.no_grad():
            with self.ldm_model.ema_scope():

                # encode (scaled latent)
                z_enc = self.ldm_sampler.stochastic_encode(
                    z, torch.tensor([self.t_enc]*self.batch_size).to(self.device)
                )
                
                samples = self.ldm_sampler.decode(z_enc, c, self.t_enc, unconditional_guidance_scale=self.scale,
                                    unconditional_conditioning=self.uc)
        return samples
    

    def validate(self):
        self.direction_model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, x in enumerate(self.val_loader):
                loss = self.__get_batch_loss(x)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)
    

    def __load_model_from_config(self):
        ckpt = self.configs['model']['ldm_model']['ckpt']
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(self.configs['model']['ldm_model'])
        m, u = model.load_state_dict(sd, strict=False)

        model.to(self.device)
        model.eval()
        return model



if __name__ == "__main__":
    trainer = DirectionModelTrainer(OmegaConf.load('dl_project/configs/direction_model_training.yaml')['training'])
    trainer.train()


