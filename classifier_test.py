import sys

import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from models import SimpleClassifier
from UNet import UNet
from torchinfo import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# transform_train = torch.nn.Sequential(
#     transforms.RandomResizedCrop(size=(32, 32), antialias=True),
#     transforms.RandomPerspective(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
#
# transform_test = torch.nn.Sequential(
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
#
# scripted_transforms_train = torch.jit.script(transform_train)
# scripted_transforms_test = torch.jit.script(transform_test)

transform_train = transforms.Compose([transforms.ToTensor(),
                                      transforms.RandomResizedCrop(size=(32, 32), antialias=True),
                                      transforms.RandomPerspective(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

scripted_transforms_train = transform_train
scripted_transforms_test = transform_test

batch_size = 128

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=scripted_transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1, pin_memory=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=scripted_transforms_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1, pin_memory=True)


super_classes = [
    'aquatic mammals',
    'fish',
    'flowers',
    'food containers',
    'fruit and vegetables',
    'household electrical devices',
    'household furniture',
    'insects',
    'large carnivores',
    'large man-made outdoor things',
    'large natural outdoor scenes',
    'large omnivores and herbivores',
    'medium-sized mammals',
    'non-insect invertebrates',
    'people',
    'reptiles',
    'small mammals',
    'trees',
    'vehicles 1',
    'vehicles 2']

classes = [
    'beaver', 'dolphin', 'otter', 'seal', 'whale',
    'aquarium fish', ' flatfish', ' ray', ' shark', ' trout',
    'orchids', ' poppies', 'roses', 'sunflowers', 'tulips',
    'bottles', 'bowls', 'cans', 'cups', 'plates',
    'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
    'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
    'bed', 'chair', 'couch', 'table', 'wardrobe',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'bear', 'leopard', 'lion', 'tiger', 'wolf',
    'bridge', 'castle', 'house', 'road', 'skyscraper',
    'cloud', 'forest', 'mountain', 'plain', 'sea',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
    'crab', 'lobster', 'snail', 'spider', 'worm',
    'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    'maple', 'oak', 'palm', 'pine', 'willow',
    'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
    'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


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
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(len(trainloader))
    net = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
    summary(net)

    opt_net = torch.compile(net)
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
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/100:.3f}')
                running_loss = 0.0
        evaluate(opt_net, trainloader)
        evaluate(opt_net, testloader)
        #scheduler1.step()
    print('Finished Training')
