import torch
try:
    from net_utils import create_mlp
    from abstract import Model
except:
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
        **kwargs
    ):

        super().__init__(k=direction_count, size=c_length, alpha=alpha, normalize=normalize)
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



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to make it a 3D tensor
        x = torch.reshape(x, [-1, 1, self.c_length])  # shape: [batch_size, 1, c_length]

        # Repeat along the second dimension for each direction
        x = x.repeat(1, self.direction_count, 1)  # shape: [batch_size, self.k, c_length]

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

        dz = torch.cat(dz, dim=1)  # Concatenating along the direction dimension

        # Add directions
        x = x + dz  # shape: [batch_size, self.direction_count, c_length]

        return x  # The output shape is [batch_size, self.direction_count, c_length]

    def forward_single(self, x: torch.Tensor, direction_index: int) -> torch.Tensor:
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

        return transformed_x