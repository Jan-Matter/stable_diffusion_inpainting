from typing import Tuple, Any
import torch
from torch import Tensor
from dl_project.direction_models.net_utils import create_mlp
from dl_project.direction_models.abstract import Model
from dl_project.direction_models.direction_model import DirectionModel
from dl_project.loss.contrastive_loss import *
from latentclr.colat.generators import BigGANGenerator
from latentclr.colat.projectors.identity import IdentityProjector


def main():

    # Parameters
    batch_size = 5  # Number of vectors in the batch
    c_length = 128  # Length of each vector
    direction_count = 7  # Number of directions
    depth = 2  # Depth of MLPs in DirectionModel

    # Generate a batch of arbitrary vectors
    x = torch.randn(batch_size, c_length)

    # Instantiate the DirectionModel
    model = DirectionModel(
        direction_count=direction_count,
        c_length=c_length,
        depth=depth,
        alpha=0.1,
        normalize=True,
        bias=True,
        batchnorm=True,
        final_norm=False
    )

    # replace generator and projector using those for diffusion models
    """
    Install additional packages using the following commands:
    pip install pytorch-pretrained-biggan==0.1.1
    pip install nltk==3.5
    """
    dummy_generator = BigGANGenerator(resolution='256', device=x.device, truncation=0.4, class_name='bulbul', feature_layer='generator.layers.4')
    dummy_projector = IdentityProjector(normalize=True)
    ct_loss = ContrastiveLoss(generator=dummy_generator, projector=dummy_projector, model=model, K=direction_count)
    acc, loss = ct_loss.contrastive_loss(x)

    # Print the output from the regular forward method
    print("Accuracy:    ", acc.item())
    print("Loss:        ", loss.item())


if __name__ == "__main__":
    main()
