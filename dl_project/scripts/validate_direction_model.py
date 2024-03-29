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
from ldm.models.diffusion.ddim import DDIMSampler

from dl_project.direction_models.direction_model import DirectionModel
from dl_project.loss.contrastive_loss import ContrastiveLoss
from dl_project.datasets.image_caption_dataset import ImageCaptionDataset
from dl_project.datasets.custom_caption_dataset import CustomImageCaptionDataset

class DirectionModelValidator:
     
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device(configs['device'])
        self.model_save_path = configs['model']['direction_model']['path']
        self.direction_count = configs['model']['direction_model']['direction_count']
        self.direction_model = DirectionModel(**configs['model']['direction_model'])
        self.ldm_model = self.__load_model_from_config(configs['model']['ldm_model'])
        self.ldm_sampler = DDIMSampler(self.ldm_model)
        self.ddim_steps = configs['ddim_steps']
        self.ddim_eta = configs['ddim_eta']
        self.ldm_sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)
        self.loss = ContrastiveLoss(**configs['loss'])
        self.optimizer = torch.optim.Adam(self.direction_model.parameters(), lr=configs['lr'])
        self.direction_model.to(self.device)

        #dataset = ImageCaptionDataset(configs['dataset'], training=False)
        dataset = CustomImageCaptionDataset(configs['dataset'], training=False)
        dataset = torch.utils.data.Subset(dataset, range(30))
        self.batch_size = configs['batch_size']
        self.val_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.best_loss = None

        #defaults: strength = 0.8 and ddim_steps = 50 and scale = 5.0
        self.t_enc = int(configs['strength'] * configs['ddim_steps'])
        self.scale = configs['scale']
        self.uc = self.ldm_model.get_learned_conditioning([""]).requires_grad_(True).to(self.device)



    
    def validate(self):
        self.direction_model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, x in enumerate(self.val_loader):
                loss = self.__get_batch_loss(x)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)
    

    def __get_batch_loss(self, x):
        images = x['image'].to(self.device)
        captions = x['caption']
        captions_enc = [self.ldm_model.get_learned_conditioning([caption]) for caption in captions]
        #since all captions are the same, we can just take the first
        caption_enc = captions_enc[0]
        #since the caption has only a length of 5 we can take the first 5 tokens as latent space
        token_count = self.configs['model']['direction_model']['c_length'] // 768
        caption_enc = caption_enc[:, :token_count, :].repeat(self.batch_size, 1, 1)
        caption_enc_reshaped = caption_enc.reshape(self.batch_size, -1)

        noised_images_enc_orig = self.ldm_model.get_first_stage_encoding(
            self.ldm_model.encode_first_stage(images)
        )

        # [batch_size, noised_image_enc_length] -> [batch_size, direction_count, noised_image_enc_length]
        noised_images_enc = noised_images_enc_orig.reshape(self.batch_size, 1, -1)
        noised_images_enc = noised_images_enc.repeat(1, self.direction_count, 1)
        noised_images_enc = noised_images_enc.reshape(self.batch_size * self.direction_count, 4, 32, 32)
        
        # [batch_size, caption_enc_length] -> [batch_size, direction_count, caption_enc_length]
        directions = self.direction_model(caption_enc_reshaped)
        directions = directions.reshape(self.batch_size, self.direction_count, token_count, -1)
        captions_enc_rest = captions_enc[0]\
            .unsqueeze(1)\
            .repeat(self.batch_size, self.direction_count, 1, 1)\
                [:, :, token_count:, :]

        directions = torch.cat([directions, captions_enc_rest], dim=2)

        orig_caption_enc = captions_enc[0]
        
        # apply stable diffusion model to input images
        rand1 = random.randint(0, self.batch_size - 1)
        rand2 = random.randint(0, self.direction_count - 1)
        features = []
        for batch_idx in range(self.batch_size):
            batch_features = []
            with torch.no_grad():
                noised_image_enc_orig = noised_images_enc_orig[batch_idx]
                noised_image_enc_orig = noised_image_enc_orig.unsqueeze(0)
                orig_decoded_samples = self.__decode(noised_image_enc_orig, orig_caption_enc)
            for direction_idx in range(self.direction_count):
                noised_image_enc = noised_images_enc[batch_idx * self.direction_count + direction_idx]
                noised_image_enc = noised_image_enc.unsqueeze(0)
                direction = directions[batch_idx, direction_idx]
                direction = direction.unsqueeze(0)

                if rand1 % self.batch_size != batch_idx or self.run2 % self.direction_count != direction_idx:
                    # this is necessary to keep the memory usage in bounds
                    with torch.no_grad():
                        decoded_samples = self.__decode(noised_image_enc, direction)
                        decoded_samples = decoded_samples.detach()
                else:
                    decoded_samples = self.__decode(noised_image_enc, direction)

                batch_features.append(torch.sub(decoded_samples, orig_decoded_samples))
            batch_features = torch.stack(batch_features)
            features.append(batch_features)
        features = torch.stack(features)
        
        # reshape features to [batch_size * direction_count, feature_length].
        reshaped_features = features.reshape(
            features.shape[0] * features.shape[1], -1
        )
            
        # 4. create group_indices of shape[batch_size * direction_count]: which feature is from which direction model.
        group_indices = torch.stack([torch.ones(features.shape[0]) * i
                                        for i in range(features.shape[1])])\
                                        .T\
                                        .reshape(features.shape[0] * features.shape[1])

        loss = self.loss(reshaped_features, group_indices)
        return loss

    def __decode(self, z, c):
        with self.ldm_model.ema_scope():

            # encode (scaled latent)
            z_enc = self.ldm_sampler.stochastic_encode(
                z,
                torch.tensor([self.t_enc]).to(self.device)
            )
            samples = self.ldm_sampler.decode(z_enc, c, self.t_enc, unconditional_guidance_scale=self.scale,
                                unconditional_conditioning=self.uc)
        return samples
    

    def __load_model_from_config(self, configs):
        ckpt = configs['ckpt']
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
    validator = DirectionModelValidator(OmegaConf.load('dl_project/configs/direction_model_training.yaml')['training'])
    print(validator.validate())