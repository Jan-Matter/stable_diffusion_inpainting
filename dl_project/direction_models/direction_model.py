from torch import nn


class DirectionModel(nn.Module):

    def __init__(self, c_length, direction_count):
        super(DirectionModel, self).__init__()
        self.c_length = c_length
        self.direction_count = direction_count
        
    
    def forward(self, x):
        """
        Args:
            x: torch.tensor(batch_size, c_length)
        Returns:
            torch.tensor(batch_size, direction_count, c_length)
        """
        pass