import torch

class ContrastiveLoss:

    def __init__(self, direction_count, latent_diff_dim, ABS=True, TEMP=0.5, K=None, REDUCE='mean'):

        self.abs = ABS
        self.temp = TEMP
        self.k = K
        self.reduce = REDUCE

    def contrastive_loss_using_differences(self, dz):

        # prepare features
        features = []
        for i in range(dz.shape[1]):
            dz_batch = dz[:, i, :]
            feats = dz_batch
            feats = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))
            features.append(feats)
        features = torch.cat(features, dim=0)

        # Start computing accuracy and loss
        out = features
        n_samples = len(out)
        assert (
                           n_samples % self.k) == 0, "Batch size is not divisible by given k!"

        # similarity matrix
        sim = torch.mm(out, out.t().contiguous())

        if self.abs:
            sim = torch.abs(sim)

        sim = torch.exp(sim * self.temp)

        # mask for pairs
        mask = torch.zeros((n_samples, n_samples), device=sim.device).bool()
        for i in range(self.k):
            start, end = i * (n_samples // self.k), (i + 1) * (
                        n_samples // self.k)
            mask[start:end, start:end] = 1

        diag_mask = ~(torch.eye(n_samples, device=sim.device).bool())

        # pos and neg similarity and remove self similarity for pos
        pos = sim.masked_select(mask * diag_mask).view(n_samples, -1)
        neg = sim.masked_select(~mask).view(n_samples, -1)

        if self.reduce == "mean":
            pos = pos.mean(dim=-1)
            neg = neg.mean(dim=-1)
        elif self.reduce == "sum":
            pos = pos.sum(dim=-1)
            neg = neg.sum(dim=-1)
        else:
            raise ValueError("Only mean and sum is supported for reduce method")

        acc = (pos > neg).float().mean()
        loss = -torch.log(pos / neg).mean()
        return acc, loss




if __name__ == "__main__":

    # test contrastive loss

    from typing import Tuple, Any
    import torch
    from torch import Tensor
    from dl_project.direction_models.net_utils import create_mlp
    from dl_project.direction_models.abstract import Model


    class DirectionModel(Model):

        def __init__(
                self,
                direction_count: int,
                c_length: int,
                depth: int,
                alpha: float = 0.1,
                normalize: bool = True,
                bias: bool = True,
                batchnorm: bool = True,
                final_norm: bool = False,
        ):

            super().__init__(k=direction_count, size=c_length, alpha=alpha,
                             normalize=normalize)
            # super(DirectionModel, self).__init__()
            self.c_length = c_length
            self.direction_count = direction_count

            # make mlp net
            self.nets = torch.nn.ModuleList()

            for i in range(direction_count):
                net = create_mlp(
                    depth=depth,
                    in_features=c_length,
                    middle_features=c_length,
                    out_features=c_length,
                    bias=bias,
                    batchnorm=batchnorm,
                    final_norm=final_norm,
                )
                self.nets.append(net)

        def forward(self, x: torch.Tensor) -> tuple[Any, Tensor]:
            # Reshape to make it a 3D tensor
            x = torch.reshape(x, [-1, 1,
                                  self.c_length])  # shape: [batch_size, 1, c_length]

            # Repeat along the second dimension for each direction
            x = x.repeat(1, self.direction_count,
                         1)  # shape: [batch_size, self.k, c_length]

            # Calculate directions
            dz = []
            for i in range(self.direction_count):
                # Selecting the slice for the i-th direction across all batch elements
                slice_x = x[:, i, :]

                res_dz = self.nets[i](slice_x)
                res_dz = self.post_process(res_dz)
                # Add a direction dimension to res_dz
                res_dz = res_dz.unsqueeze(1)  # [batch_size, 1, c_length]
                dz.append(res_dz)

            dz = torch.cat(dz,
                           dim=1)  # Concatenating along the direction dimension

            # Add directions
            x = x + dz  # shape: [batch_size, self.direction_count, c_length]

            return x, dz  # The output shape is [batch_size, self.direction_count, c_length]

        def forward_single(self, x: torch.Tensor,
                           direction_index: int) -> torch.Tensor:
            """
            Apply a single direction's transformation to the input tensor.

            Args:
            x (torch.Tensor): Input tensor.
            direction_index (int): Index of the direction to apply.

            Returns:
            torch.Tensor: Transformed tensor.
            """
            if direction_index < 0 or direction_index >= self.direction_count:
                raise ValueError("direction_index is out of bounds")

            x = x.view(-1, self.c_length)  # Reshape x to [batch_size, c_length]

            # Apply the transformation of the specified direction
            transformed_x = self.nets[direction_index](x)
            transformed_x = self.post_process(transformed_x)

            return transformed_x # The output shape is [batch_size, self.direction_count, c_length]


    # Parameters
    batch_size = 5  # Number of vectors in the batch
    c_length = 10  # Length of each vector
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

    # Run the batch through the model using the regular forward method
    output, dz = model(x)

    ct_loss = ContrastiveLoss(None, None, K=dz.shape[1])
    acc, loss = ct_loss.contrastive_loss_using_differences(dz)

    # Print the output from the regular forward method
    print("Accuracy:    ", acc.item())
    print("Loss:        ", loss.item())
