import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import json
using_gpu = torch.cuda.is_available()

def loading_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    valid_test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    return trainloader, testloader, validloader, train_data


# Build and train your network
# Freeze parameters so we don't backprop through them
def make_model(arch, hidden_units, lr):
    if arch=="densenet121":
        model = models.densenet121(pretrained=True)
        input_size = 1024
    else:
        model = models.vgg16(pretrained=True)
        input_size = 25088
    output_size = 102
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, output_size)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    return model, input_size


# Training the model
def train_model(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device):
    epochs = epochs
    print_every = print_every
    steps = 0

    if device=='gpu':
        model = model.to('cuda')

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                    f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0


# TODO validation on the test set
def test_model(model, testloader):
    model.eval()
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            logps = model(inputs)

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy/len(testloader):.3f}")



# Save the checkpoint
def save_checkpoints(model, arch, lr, epochs, input_size, hidden_units, class_to_idx, checkpoint_path):
    model.class_to_idx = class_to_idx
    checkpoint = {
            'structure' :arch,
            'learning_rate': lr,
            'epochs': epochs,
            'input_size': input_size,
            'hidden_units':hidden_units,
            'state_dict':model.state_dict(),
            'class_to_idx': model.class_to_idx
        }
    torch.save(checkpoint, checkpoint_path + 'checkpoint.pth')
    print('Training checkpoint has been saved in ', checkpoint_path + 'checkpoint.pth')


# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(path):
    checkpoint = torch.load(path)
    lr = checkpoint['learning_rate']
    input_size = checkpoint['input_size']
    structure = checkpoint['structure']
    hidden_units = checkpoint['hidden_units']
    epochs = checkpoint['epochs']

    model,_ = make_model(structure, hidden_units, lr)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    pil_image = Image.open(image)

    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = image_transforms(pil_image)
    return img

# Labeling
def labeling(flower_names):
    with open(flower_names, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name


# Class Prediction
def predict(processed_image, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file
    model.eval()
    model.cpu()
    processed_image = processed_image.unsqueeze_(0)
    processed_image = processed_image.float()

    with torch.no_grad():
        output = model.forward(processed_image)
        probs, classes = torch.topk(input=output, k=topk)
        top_prob = probs.exp()

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in classes.cpu().numpy()[0]]

    return top_prob, top_classes









# Training *************************

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from PIL import Image
import argparse
import utility_functions


# Argparser Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-p', action='store', dest='path', help='path of directory', required=True)
parser.add_argument('--save_dir', action='store', dest='cp_path', default='.', help='path of checkpoint')
parser.add_argument('--arch', action='store', dest='arch', default='densenet121', choices={"densenet121", "vgg16"}, help='architecture of the network')
parser.add_argument('--learning_rate', action='store', nargs='?', default=0.002, type=float, dest='learning_rate', help='(float) learning rate of the network')
parser.add_argument('--epochs', action='store', dest='epochs', default=1, type=int, help='(int) number of epochs while training')
parser.add_argument('--hidden_units', action='store', nargs=1, default=500, dest='hidden_units', type=int,
                    help='Enter hidden units of the network')
parser.add_argument('--gpu', action='store_true', default=False, dest='boolean_t', help='Set a switch to use GPU')
results = parser.parse_args()


data_dir = results.path
checkpoint_path = results.cp_path
arch = results.arch
hidden_units = results.hidden_units
epochs = results.epochs
lr = results.learning_rate
gpu = results.boolean_t
print_every = 20

if gpu==True:
    using_gpu = torch.cuda.is_available()
    device = 'gpu'
    print('GPU On');
else:
    print('GPU Off');
    device = 'cpu'


# Loading Dataset
trainloader, testloader, validloader, train_data  = utility_functions.loading_data(data_dir)
class_to_idx = train_data.class_to_idx

# Network Setup
model, input_size = utility_functions.make_model(arch, hidden_units, lr)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

# Training Model
utility_functions.train_model(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device)

# Testing Model
utility_functions.test_model(model, testloader)

# Saving Checkpoint
utility_functions.save_checkpoints(model, arch, lr, epochs, input_size, hidden_units, class_to_idx, checkpoint_path)







# Predict *************************
# Imports here

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
import json
import time
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
import seaborn as sns
from workspace_utils import active_session
import argparse
import utility_functions

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', action='store', dest='img_path', help='path of image to predict', required=True)
parser.add_argument('--checkpoint_path', action='store', dest='checkpoint_path', help='path of checkpoint', required=True)
parser.add_argument('--top_k', action="store", default=5, dest="top_k",  type=int)
parser.add_argument('--flower_names', action="store", dest="flower_names", default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', default=False, dest='switch', help='Set a switch to use GPU')
results = parser.parse_args()

img_path = results.img_path
checkpoint_path = results.checkpoint_path
top_k = results.top_k
flower_names = results.flower_names
gpu = results.switch

if gpu==True:
    using_gpu = torch.cuda.is_available()
    device = 'gpu'
    print('gpu On');
else:
    print('gpu Off');
    device = 'cpu'

model = utility_functions.load_checkpoint(checkpoint_path)
processed_image = utility_functions.process_image(img_path)
probs, classes = utility_functions.predict(processed_image, model, top_k, device)
# Label mapping
cat_to_name = utility_functions.labeling(flower_names)

labels = []
for class_index in classes:
    labels.append(cat_to_name[str(class_index)])

# Converting from tensor to numpy-array
print('Name of class: ', labels)
print('Probability: ', probs)
