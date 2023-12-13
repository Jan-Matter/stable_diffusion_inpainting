

class ContrastiveLoss:

    def __init__(self, direction_count, latent_diff_dim):
        self.direction_count = direction_count
        self.latent_diff_dim = latent_diff_dim

    def contrastive_loss(y_direction1, y_direction2):
        """
        Args:
            y_direction1: torch.tensor(batch_size, direction_count, latent_diff_dim)
            y_direction2: torch.tensor(batch_size, direction_count, latent_diff_dim)
        Returns:
            loss: torch.tensor(batch_size, ) (1d tensor of batch_size length)
        """
        """
        """