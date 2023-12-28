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
        self.direction_models = [DirectionModel(**configs['model']['direction_model']) for _ in range(configs['num_direction_models'])]
        self.ldm_model = self.__load_model_from_config(configs['model']['ldm_model'])
        self.ldm_sampler = PLMSSampler(self.ldm_model)
        self.loss = ContrastiveLoss(**configs['loss'])
        self.optimizers = [torch.optim.Adam(model, lr=configs['lr']) for model in self.direction_models]
        self.device = torch.device(configs['device'])
        self.model.to(self.device)

        dataset = ImageCaptionDataset(configs['dataset'])
        train_size = int(configs['train_size'] * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
        self.batch_size = configs['batch_size']
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.epochs = configs['epochs']

        #defaults: strength = 0.8 and ddim_steps = 50 and scale = 5.0
        self.t_enc = int(configs['strength'] * configs['ddim_steps'])
        self.scale = configs['scale']
        self.uc = self.ldm_model.get_learned_conditioning(self.batch_size * [""])

        seed_everything(configs['seed'])


    
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            for batch_idx, x in enumerate(self.train_loader):
                trained_model_idx = random.randint(0, len(self.direction_models) - 1)
                self.optimizers[trained_model_idx].zero_grad()

                images, captions = images.to(self.device), captions.to(self.device)
                captions_enc = self.ldm_model.get_learned_conditioning(captions)
                noised_images_enc = self.ldm_model.get_first_stage_encoding(
                    self.ldm_model.encode_first_stage(images)
                )
                
                trained_direction_c_ground = self.direction_models[trained_model_idx](captions_enc)

                with torch.no_grad():
                    trained_direction_c_same = self.direction_models[trained_model_idx](captions_enc)
                    trained_direction_c_diff = self.direction_models[trained_model_idx](captions_enc)

                decoded_latent_ground = data_parallel(self.__decode, (noised_images_enc, trained_direction_c_ground), dim=-1)
                decoded_latent_same = data_parallel(self.__decode, (noised_images_enc, trained_direction_c_ground), dim=-1)
                decoded_latent_diff = data_parallel(self.__decode, (noised_images_enc, trained_direction_c_ground), dim=-1)
                
                loss_same = self.loss.contrastive_loss(decoded_latent_ground, decoded_latent_same)
                loss_diff = self.loss.contrastive_loss(decoded_latent_ground, decoded_latent_diff)
                loss = loss_same - loss_diff
                loss.backward()
                self.optimizers[trained_model_idx].step()
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
            self.eval(self.val_loader)
            torch.save(self.model.state_dict(), f'models/direction_model_{epoch}.pt')
    
    def __decode(self, z, c):
        with torch.no_grad():
            with self.ldm_model.ema_scope():

                # encode (scaled latent)
                z_enc = self.ldm_sampler.stochastic_encode(
                    z, torch.tensor([self.t_enc]*self.batch_size).to(self.device)
                )
                
                samples = self.ldm_sampler.decode(z_enc, c, self.t_enc, unconditional_guidance_scale=self.scale,
                                    unconditional_conditioning=self.uc)
                x_sample = self.ldm_model.decode_first_stage(samples)[0]
                x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        return x_sample
                    

    def __load_img(self, image):
        #TODO add transform to dataset
        #image = Image.open(path).convert("RGB")
        w, h = image.size
        print(f"loaded input image of size ({w}, {h}) from {path}")
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0
    

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
    trainer = DirectionModelTrainer(OmegaConf.load('dl_project/configs/direction_model_trainging.yaml')['training'])
    trainer.train()


