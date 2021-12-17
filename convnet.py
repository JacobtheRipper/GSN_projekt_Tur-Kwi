import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimisation algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches
# import torchvision.datasets as datasets # Has standard datasets that can be imported more easily
import torchvision.transforms as transforms # Transformations that can be performed on the dataset
from customDataset import customDataset

# Hyperparameters
input_channels = 1
num_classes = 8
learning_rate = 0.001
batch_size = 64
num_epochs = 35
load_model = False  # enables loading from a given checkpoint
load_from_epoch = 20  # loads model that has been trained for 'load_from_epoch' epochs

# Convolutional Neural Network model based on the architecture found in
# "Exploring Data Augmentation to Improve Music Genre Classification with ConvNets" article
# the input of this CNN is a grayscale 64x192 image containing mel spectrogram

# TODO improve model by adding SVM at the end
class ConvNet(nn.Module):
    def __init__(self, input_channels=input_channels, num_classes=num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=(5, 5), stride=1, padding=2)  # output's shape is equal to its input shape
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # downsampling from 64x192 to 32x96 or from 32x96 to 16x48
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=1, padding=2)  # same as conv1
        self.fc1 = nn.Linear(in_features=16*48*1, out_features=500)  # in_features = image_size*out_channels
        self.fc2 = nn.Linear(in_features=500, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = F.dropout(F.relu(x), p=0.5, training=True, inplace=False)
        x = self.fc2(x)
        return x

# Test code to check the size of the output tensor
x = torch.randn(64, 1, 64, 192)
model = ConvNet()
print(model(x).shape)

# Function for evaluation
def check_accuracy(loader, model):
    if loader == train_data_loader:
        print("Checking accuracy on training data")
    elif loader == validation_data_loader:
        print("Checking accuracy on validation data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    print(f'Correctly classified examples: {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

def save_checkpoint(state, current_epoch):
    print(f"===> Saving checkpoint at epoch: {current_epoch}")
    filename = "convnet_checkpoint_epoch_" + str(current_epoch) + ".pt"
    torch.save(state, filename)

def load_checkpoint(starting_epoch):
    print(f"===> Loading checkpoint at epoch: {starting_epoch}")
    filename = "convnet_checkpoint_epoch_" + str(starting_epoch) + ".pt"
    state = torch.load(filename)
    model.load_state_dict(state['state_dict'])
    optimiser.load_state_dict(state['optimizer'])

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialise model
model = ConvNet().to(device)

# Load data from MNIST dataset
# TODO remove this code
'''
composed_transform = transforms.Compose([transforms.Resize(size=[256, 16]), transforms.ToTensor()])

train_dataset = datasets.MNIST(root="dataset/", train=True, transform=composed_transform, download=True)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", train=False, transform=composed_transform, download=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
'''

# Load custom data from modified FMA small dataset
# TODO make sure to use correct data directories

composed_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=[64, 192]), transforms.ToTensor()])

CSV_DIR = "dataset/content/fma/FMA_spectrograms/data_annot.csv"
SPECTR_DIR = "dataset/content/fma"

dataset = customDataset(annotation_file=CSV_DIR, data_dir=SPECTR_DIR, transform=composed_transform)
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [3576, 447, 447])  # using 80/10/10% (training/validation/test) ratio

train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# Loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# Train CNN

# optional model loading from a .pt file. A correct filename should be used

if load_model:
    load_checkpoint(load_from_epoch)


for epochs in range(num_epochs):
    print(f"Epoch {epochs+1} out of {num_epochs} ")
    for batch_idx, (data, targets) in enumerate(train_data_loader):
        #Get data to Cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        #forward propagation
        scores = model(data)
        loss = criterion(scores, targets)

        #backward propagation
        optimiser.zero_grad()
        loss.backward()  # may crash if the number of target labels is greater than the number of classes

        #gradient descent
        optimiser.step()
    # print(loss)  # optional line for testing the CNN
    
    # optional checkpoint saving every 5 epochs
    if (epochs+1) % 5 == 0:
        model_state = {'state_dict': model.state_dict(), 'optimizer': optimiser.state_dict()}
        save_checkpoint(state=model_state, current_epoch=epochs+1)

        # Check accuracy on training, validation & test data
        check_accuracy(train_data_loader, model)
        check_accuracy(validation_data_loader, model)
        check_accuracy(test_data_loader, model)
