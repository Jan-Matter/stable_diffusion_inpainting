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
    num_features = 8  # Number of vectors in the batch
    feature_length = 128  # Length of each vector
    group_indices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int)
    features = torch.randn(num_features, feature_length).cuda()

    ct_loss = ContrastiveLoss()

    loss = ct_loss(features, group_indices, reduce='mean')
    loss_matrix = ct_loss(features, group_indices, reduce='none')

    vectorized_loss = ct_loss.vectorized_forward(features, group_indices, reduce='mean')
    vectorized_loss_matrix = ct_loss.vectorized_forward(features, group_indices, reduce='none')

    print(loss)
    print(loss_matrix)
    print(vectorized_loss)
    print(vectorized_loss_matrix)

    print("loss and vectorized loss are the same:", (loss.item() - vectorized_loss.item()) < 1e-4)
    print("loss matrix and vectorized loss matrix are the same:", (torch.abs(loss_matrix - vectorized_loss_matrix)).max().item() < 1e-4)



if __name__ == "__main__":
    main()

