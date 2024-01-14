import argparse, os
import torch
from  torch.nn.parallel import data_parallel
import numpy as np
import PIL
from omegaconf import OmegaConf
from PIL import Image
from imwatermark import WatermarkEncoder
from pytorch_lightning import seed_everything
from einops import rearrange
import random

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

from dl_project.direction_models.direction_model import DirectionModel
from dl_project.datasets.image_caption_dataset import ImageCaptionDataset
from dl_project.datasets.custom_caption_dataset import CustomImageCaptionDataset

class DirectionModelInference:
     
    def __init__(self, configs):
        self.configs = configs
        self.device = torch.device(configs['device'])
        self.model_save_path = configs['model']['direction_model']['path']
        self.direction_count = configs['model']['direction_model']['direction_count']
        self.direction_model = DirectionModel(**configs['model']['direction_model'])
        self.direction_model.load_state_dict(torch.load(configs['model']['direction_model']['path']))
        self.ldm_model = self.__load_model_from_config(configs['model']['ldm_model'])
        self.decoded_output_path = configs['decoded_output_path']
        self.ldm_sampler = DDIMSampler(self.ldm_model)
        self.ddim_steps = configs['ddim_steps']
        self.ddim_eta = configs['ddim_eta']
        self.ldm_sampler.make_schedule(ddim_num_steps=self.ddim_steps, ddim_eta=self.ddim_eta, verbose=False)
        self.direction_model.to(self.device).half()

        if configs['dataset']['name'] == "maskara":
            dataset = CustomImageCaptionDataset(configs['dataset'], training=False)
        else:
            dataset = ImageCaptionDataset(configs['dataset'], training=False)
        self.dataset = torch.utils.data.Subset(dataset, range(100))

        #defaults: strength = 0.8 and ddim_steps = 50 and scale = 5.0
        self.t_enc = int(configs['strength'] * configs['ddim_steps'])
        self.scale = configs['scale']
        self.uc = self.ldm_model.get_learned_conditioning([""]).requires_grad_(True).to(self.device)

        seed_everything(configs['seed'])


    def edit(self, idx, caption=None):
        x = self.dataset[idx]
        image = x['image'].to(self.device).unsqueeze(0)
        caption = x['caption']
        #caption = ["Apply lipstick to lips."]
        #caption = [""]
        if caption is not None:
            caption = [caption]

        orig_caption_enc = self.ldm_model.get_learned_conditioning(caption)
        token_count = self.configs['model']['direction_model']['c_length'] // 768
        batch_size = 3 #only used to be compatible with direction model
        caption_enc = orig_caption_enc[:, :token_count, :].repeat(batch_size, 1, 1)
        caption_enc_reshaped = caption_enc.reshape(batch_size, -1)


        noised_images_enc = self.ldm_model.get_first_stage_encoding(
            self.ldm_model.encode_first_stage(image)
        )
        noised_images_enc = noised_images_enc.reshape(1, 1, -1)
        noised_images_enc = noised_images_enc.repeat(1, self.direction_count, 1)
        noised_images_enc = noised_images_enc.reshape(1 * self.direction_count, 4, 64, 64)

        
        # [batch_size, caption_enc_length] -> [batch_size, direction_count, caption_enc_length]
        caption_enc_reshaped = caption_enc_reshaped.half()
        directions = self.direction_model(caption_enc_reshaped)
        directions = directions.float()
        directions = directions.reshape(batch_size, self.direction_count, -1)
        directions = directions[0]

        directions = directions.reshape(1, self.direction_count, token_count, -1)
        captions_enc_rest = orig_caption_enc\
            .unsqueeze(1)\
            .repeat(1, self.direction_count, 1, 1)\
                [:, :, token_count:, :]
        directions_output = torch.cat([directions, captions_enc_rest], dim=2)


        for direction_idx in range(self.direction_count):
            noised_image_enc = noised_images_enc[direction_idx]
            noised_image_enc = noised_image_enc.unsqueeze(0)
            direction_output = directions_output[0, direction_idx]
            direction_output = direction_output.unsqueeze(0)

            decoded_image = self.__decode(noised_image_enc, direction_output)

            Image.fromarray(decoded_image.astype(np.uint8)).save(
                        os.path.join(self.decoded_output_path, f"image_{idx}_direction_{direction_idx}.png"))
        
        orig_decoded_image = self.__decode(noised_image_enc, orig_caption_enc)
        Image.fromarray(orig_decoded_image.astype(np.uint8)).save(
                        os.path.join(self.decoded_output_path, f"image_{idx}_orig_decoded.png"))
        
        orig_image = torch.clamp((image[0] + 1.0) / 2.0, min=0.0, max=1.0).squeeze().cpu().numpy()
        orig_image = 255. * rearrange(orig_image, 'c h w -> h w c')
        Image.fromarray(orig_image.astype(np.uint8)).save(
                        os.path.join(self.decoded_output_path, f"image_{idx}_original.png"))

    def __decode(self, z, c):
        with torch.no_grad():
             with self.ldm_model.ema_scope():

                # encode (scaled latent)
                z_enc = self.ldm_sampler.stochastic_encode(
                    z,
                    torch.tensor([self.t_enc]).to(self.device)
                )
                samples = self.ldm_sampler.decode(z_enc, c, self.t_enc, unconditional_guidance_scale=self.scale,
                                    unconditional_conditioning=self.uc)
                x_samples = self.ldm_model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                
        return x_sample
    

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
    image_decoder = DirectionModelInference(OmegaConf.load('dl_project/configs/direction_model_inference.yaml')['inference'])
    image_decoder.edit(1, caption=None)