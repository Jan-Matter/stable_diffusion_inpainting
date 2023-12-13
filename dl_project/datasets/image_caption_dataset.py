from torch.utils.data import Dataset


class ImageCaptionDataset(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        """
        Returns:
            Number of image caption pairs in the dataset
        """
        pass

    def __getitem__(self, idx):
        """
        Args:
            idx: index of the image caption pair
        Returns:
            {
                image: tensor of shape (3, 512, 512)
                caption: string should be around 5-10 words
            }
        """
        pass