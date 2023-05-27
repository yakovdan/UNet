import os
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision
import torch
import numpy as np
import shutil

from torch.utils.data import DataLoader

from dataset import CarvanaDataset

GT_IMAGES_PATH = "./data/carvana/GT/images/"
GT_MASKS_PATH = "./data/carvana/GT/masks/"
DATA_ROOT = "./data/carvana/"


def verify_images_and_masks_match(image_names, mask_names):
    image_name_only = [x.split(".")[0] for x in image_names]
    mask_names_only = [x.split(".")[0][:-5] for x in mask_names]
    if len(mask_names_only) != len(image_name_only):
        return False
    return all(map(lambda x: x[0] == x[1], zip(image_name_only, mask_names_only)))


def process_and_save_mask(input_name, output_name):
    np_image = np.pad(np.array(Image.open(input_name)), ((0, 0), (1, 1)))
    np.save(output_name, np_image)


def process_and_save_image(input_name, output_name):
    np_image = np.pad(np.array(Image.open(input_name)), ((0, 0), (1, 1), (0, 0)))
    np_image = ((np_image / 127.5) - 1).astype(np.float16)
    np.save(output_name, np_image)


def process_set(indices, kind_str, image_filenames, mask_filenames):
    for idx in indices:
        image_fname = GT_IMAGES_PATH + image_filenames[idx]
        out_image_fname = DATA_ROOT + kind_str + "_images/" + image_filenames[idx]
        #process_and_save_image(image_fname, out_image_fname)
        shutil.copy2(image_fname, out_image_fname)
        mask_fname = GT_MASKS_PATH + mask_filenames[idx]
        out_mask_fname = DATA_ROOT + kind_str + "_masks/" + mask_filenames[idx]
        #process_and_save_mask(mask_fname, out_mask_fname)
        shutil.copy2(mask_fname, out_mask_fname)


def load_gt(images_path, masks_path):
    image_filenames = sorted(os.listdir(GT_IMAGES_PATH))
    mask_filenames = sorted(os.listdir(GT_MASKS_PATH))
    if not verify_images_and_masks_match(image_filenames, mask_filenames):
        raise "Error reading GT files"

    image_indices = list(range(len(image_filenames)))
    train_ind, temp_ind = train_test_split(image_indices, test_size=0.3)
    val_ind, test_ind = train_test_split(temp_ind, test_size=1/3)
    process_set(train_ind, "train", image_filenames, mask_filenames)
    process_set(val_ind, "val", image_filenames, mask_filenames)
    process_set(test_ind, "test", image_filenames, mask_filenames)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(paths_dict, transform_dict, batch_size, pin_memory=True, num_workers=1):
    train_dataset = CarvanaDataset(paths_dict["root"] + paths_dict["train_img"],
                                   paths_dict["root"] + paths_dict["train_mask"],
                                   transform_dict["train"],
                                   )

    val_dataset = CarvanaDataset(paths_dict["root"] + paths_dict["val_img"],
                                 paths_dict["root"] + paths_dict["val_mask"],
                                 transform_dict["val"],
                                 )

    test_dataset = CarvanaDataset(paths_dict["root"] + paths_dict["test_img"],
                                  paths_dict["root"] + paths_dict["test_mask"],
                                  transform_dict["test"],
                                  )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True,
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False,
                            )
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             shuffle=False,
                             )

    return train_loader, val_loader, test_loader


def calculate_accuracy(model, dataloader):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    device = model.device

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(dataloader)}")
    return preds.detach()


if __name__ == "__main__":
    load_gt(GT_IMAGES_PATH, GT_MASKS_PATH)