import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as functional
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 dataset - 10 classes, 6000 images per class

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_classes = 10
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

training_dataset = torchvision.datasets.CIFAR10(root='.data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='.data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img/2+0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# implement conv net
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # (32-5+0)/1 +1 = 28 output size [4, 6, 28, 28]
        self.pool = nn.MaxPool2d(2, 2) #  (28-2+0)/2 +1 = 14 output size [4, 6, 14, 14]
        self.conv2 = nn.Conv2d(6, 16, 5) #  (14-5+0)/1 +1 = 10 output size [4, 16, 10, 10]
        # poll2 (10-2+0)/2 +1 = 5 output size [4, 16, 5, 5]
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x) # no softmax because it is in loss
        return x

model = ConvNet().to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#trainig rate
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4 - batch size, 3 - color channels, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'epoch: {epoch+1} / {num_epochs}, step: {i+1}/{n_total_steps}, loss = {loss.item():.4f}')

print('Finished training!')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_classes_correct = [0 for i in range(10)]
    n_classes_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # values, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if (label == pred):
                n_classes_correct[label] += 1
            n_classes_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network = {acc}')

    for i in range(10):
        acc = 100.0 * n_classes_correct[i] / n_classes_samples[i]
        print(f'Accuracy of {classes[i]}: {acc}%')
