from torch.utils.data import Dataset


class ImageCaptionDataset(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        # return the number of image caption pairs

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