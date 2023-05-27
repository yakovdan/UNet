import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from UNet import UNet
from torchinfo import summary
from carvana_utils import get_loaders

# System params
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LOAD_MODEL = False
# Hyperparameters
BATCH_SIZE = 1
NUM_EPOCHS = 100
IMAGE_HEIGHT = 1280
IMAGE_WIDTH = 1918
PIN_MEMORY = True
DATA_ROOT = "./data/carvana/"
TRAIN_IMAGES = DATA_ROOT + "train_images/"
TRAIN_MASKS = DATA_ROOT + "train_masks/"
VAL_IMAGES = DATA_ROOT + "val_images/"
VAL_MASKS = DATA_ROOT + "val_masks/"
TEST_IMAGES = DATA_ROOT + "test_images/"
TEST_MASKS = DATA_ROOT + "test_masks/"
LEARNING_RATE = 1e-4


def create_transforms():
    train_transform = A.Compose([
        A.resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255),
        ToTensorV2()
    ])

    val_transforms = A.Compose([
        A.resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255),
        ToTensorV2()
    ])

    test_transforms = A.Compose([
        A.resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255),
        ToTensorV2()
    ])

    return {"train": train_transform, 'val': val_transforms, 'test': test_transforms}

def train_one_epoch(model, dataloader, optim, loss, scaler):
    tqdmloader = tqdm(dataloader)
    for idx, (input_data, gt) in enumerate(tqdmloader):
        input_data = input_data.to(device)
        gt = gt.to(device)

        # forward pass, use AMP training
        with torch.cuda.amp.autocast:
            pred = model(input_data)
            loss_val = loss(pred, gt)

        # backprop
        optim.zero_grad()
        scaler.scale(loss_val).backward()
        scaler.step(optim)
        scaler.update()

        tqdmloader.set_postfix(loss_val=loss_val.item())





def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


if __name__ == "__main__":
    paths_dict = {
        'data_root': "./data/carvana/",
        'train_img': "train_images/",
        'train_mask': "train_masks/",
        'val_img': "val_images/",
        'val_mask': "val_masks/",
        'test_img': "test_images/",
        'test_mask': "test_masks/",
    }
    transform_dict = create_transforms()
    train_loader, val_loader, test_loader = get_loaders(paths_dict, transform_dict, BATCH_SIZE)
    net = UNet([1, 64, 128, 256, 512, 1024]).to(device)
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    summary(net)

    #opt_net = torch.compile(net)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in trange(20):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        i = 0
        for data in trainloader:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                # forward + backward + optimize
                outputs = opt_net(inputs)
                loss = criterion(outputs, labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # print statistics
            running_loss += loss.item()
            i += 1
            if i % 100 == 99:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        evaluate(opt_net, trainloader)
        evaluate(opt_net, testloader)
        # scheduler1.step()
    print('Finished Training')
