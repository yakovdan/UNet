import os
from PIL import Image
import numpy as np
from glob import glob
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        super().__init__()
        self.images_path = images_path
        self.images = sorted(os.listdir(images_path + "/*.jpg"))
        self.masks_path = masks_path
        self.masks = sorted(os.listdir(masks_path + "/*.gif"))
        self.transform = transform
        assert (CarvanaDataset.verify_images_and_masks_match(self.images, self.masks))

    @staticmethod
    def verify_images_and_masks_match(image_names, mask_names):
        image_name_only = [x.split(".")[0] for x in image_names]
        mask_names_only = [x.split(".")[0][:-5] for x in mask_names]
        if len(mask_names_only) != len(image_name_only):
            return False
        return all(map(lambda x: x[0] == x[1], zip(image_name_only, mask_names_only)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.images_path, self.images[index])
        mask_path = os.path.join(self.masks_path, self.masks[index])
        image = np.array(Image.open(image_path), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=np.float32)
        if self.transform is not None:
            augments = self.transform(image, mask)
            image, mask = augments['image'], augments["mask"]
        return image, mask
