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
parser.add_argument('--img_path', action='store',
                    dest='img_path', help='path of image to predict', required=True)
parser.add_argument('--checkpoint_path', action='store',
                    dest='checkpoint_path', help='path of checkpoint', required=True)
parser.add_argument('--top_k', action="store",
                    default=5, dest="top_k",  type=int)
parser.add_argument('--flowers_names', action="store",
                    dest="flowers_names", default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true',
                    default=False, dest='switch', help='Use or not use GPU to predict')
results = parser.parse_args()

img_path = results.img_path
checkpoint_path = results.checkpoint_path
top_k = results.top_k
flowers_names = results.flowers_names
gpu = results.switch

if gpu==True and torch.cuda.is_available():
    device = 'gpu'
    print('GPU On');
else:
    print('GPU Off');
    device = 'cpu'

model = utility_functions.load_checkpoint(checkpoint_path)
image_path = utility_functions.process_image(img_path)
probs, classes = utility_functions.predict(image_path, model, top_k, device)

# Label mapping
cat_to_name = utility_functions.labeling(flowers_names)

labels = []
for class_index in classes:
    labels.append(cat_to_name[str(class_index)])

print('Name of class: ', labels)
print('Probability: ', probs)
