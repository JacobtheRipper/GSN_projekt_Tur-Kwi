import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimisation algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # Has standard datasets that can be imported more easily
import torchvision.transforms as transforms # Transformations that can be performed on the dataset

# Convolutional Neural Network model based on the architecture found in
# "Exploring Data Augmentation to Improve Music Genre Classification with ConvNets" article
# the input of this CNN is a grayscale 256x16 image patch derived from the spectrogram

# TODO improve model by adding dropout and SVM at the end
class ConvNet(nn.Module):
    def __init(self, input_channels=1, num_classes=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=1, padding=2)  # output's shape is equal to its input shape
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # downsampling from 256x16 to 128x8 or from 128x8 to 64x4
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5), stride=1, padding=2)  # same as conv1
        self.fc1 = nn.Linear(in_features=64*4*1, out_features=500)  # in_features = image_size*out_channels
        self.fc2 = nn.Linear(in_features=500, out_features=num_classes)
    
    def forward(self, input):
        output = self.conv1(input)
        output = F.relu(output)
        output = max_pool(output)
        output = conv2(input)
        output = F.relu(output)
        output = max_pool(output)
        output = x.reshape(x.shape[0], -1)
        output = fc1(output)
        output = F.softmax(output)
        output = fc2(output)
        '''
        output = F.relu(self.conv1(input))  # raises an error
        output = self.max_pool(output)
        output = F.relu(self.conv2(output))
        output = self.max_pool(output)
        output = x.reshape(x.shape[0], -1)
        output = self.fc1(output)
        output = F.softmax(output)
        output = self.fc2(output)
        '''
        return output

#Test code to check the size of the output tensor
x = torch.randn(64, 1, 256, 16)
model = ConvNet()
print(model(x))
print(model(x).shape)
'''
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_channels = 1
num_classes = 8
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load data from dataset
# TODO write a custom data loader
train_dataset = datasets.MNIST( 
    root="dataset/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
train_data_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_dataset = datasets.MNIST(
    root="dataset/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
test_data_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True
)

# Initialise model
# TODO initialise in Google Colab
model = ConvNet().to(device)

# Loss and optimiser
criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

# Train CNN
#TODO create a train_model() function
for epochs in range(num_epochs):
	for batch_idx, (data, targets) in enumerate(train_data_loader):
		#Get data to Cuda if possible
		data = data.to(device=device)
		targets = targets.to(device=device)
        
		#forward propagation
		scores = model(data)
		loss = criterion(scores, targets)

		#backward propagation
		optimiser.zero_grad()
		loss.backward()

        	#gradient descent
		optimiser.step()
        
# Check accuracy on training & test data

def check_accuracy(loader, model):
	if loader.dataset.train:
		print("Checking accuracy on training data")
	else:
		print("Checking accuracy on test data")

	num_correct = 0
	num_samples = 0
	model.eval()

	with torch.no_grad():
		for x, y in loader:
			x = x.to(device=device)
			y = y.to(device=device)
			x = x.reshape(x.shape[0], -1)

			scores = model(x)
			_, predictions = scores.max(1)
			num_correct += (predictions == y).sum()
			num_samples += predictions.size(0)

	print(f'Correctly classified examples: {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

	model.train()

check_accuracy(train_data_loader, model)
check_accuracy(test_data_loader, model)
'''
