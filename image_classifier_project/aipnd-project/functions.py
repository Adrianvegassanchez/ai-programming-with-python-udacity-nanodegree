from data_utils import *
from model_utils import validate_model
from math import floor

import torch
import matplotlib.pyplot as plt
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms, utils
from collections import OrderedDict
from torch.autograd import Variable 
from PIL import Image

training_data = 0

def load_data(data_dir):
    
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    global training_data
    
    training_data = load_dataset(train_dir, make_train_transforms())
    testing_data = load_dataset(test_dir, make_test_transforms())
    validation_data = load_dataset(valid_dir, make_validation_transforms())
    
    train_dataloader = make_dataloaders(training_data)
    test_dataloader = make_dataloaders(testing_data)
    validation_dataloader = make_dataloaders(validation_data)
    
    print("Training info: ")
    print_dataset_info(training_data)
    print("Testing info: ")
    print_dataset_info(testing_data)
    print("Validation info: ")
    print_dataset_info(validation_data)
    
    return train_dataloader, test_dataloader, validation_dataloader


def create_model(arch, hidden_units):
    
    global training_data
        
    arch = arch.lower()
    is_valid, model, input_dimension = validate_model(arch)
        
    if is_valid:
        
        for param in model.parameters():
            param.requires_grad = False
                            
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_dimension, hidden_units)),
            ('relu', nn.ReLU()),
            ('drpot', nn.Dropout(p=0.05)),
            ('fc2', nn.Linear(hidden_units, len(training_data.classes))),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        
        print("Hidden units : {}".format(hidden_units))
        
    return model

def train(model, trainloader, validationloader, learning_rate, use_gpu, epochs=5, print_every=40):
    
    steps = 0
    running_loss = 0
    
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    criterion = nn.NLLLoss()
    

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        # Model in training mode
        model.train()
        
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, epochs))
    
        # Iterate over data
        for inputs, labels in trainloader:
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            
            output = model.forward(inputs)
            loss = criterion(output, labels)
            
            #backward + optimize 
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:

                #Model in inference mode
                model.eval()
                                
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, criterion, validationloader, device)
                    
                print('Training loss: {:.4f}'.format(running_loss/print_every))
                print('Validation Loss: {:.4f}'.format(validation_loss))
                print('Validation Accuracy: {:.4f}'.format(accuracy))
            
                running_loss = 0
                model.train()
                
    return model, optimizer, criterion
                    
def validation(model, criterion, data_loader, device):
    
    accuracy = 0
    valid_loss = 0
    
    #Iterate over data from valid dataLoader
    for inputs, labels in data_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        
        #Calculate loss
        valid_loss += criterion(output, labels).item()
        
        #Calculate probability 
        ps = torch.exp(output)
        
        #Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        
        #number of correct predictions divided by all predictions
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    #return test_loss/len(data_loader), accuracy/len(data_loader)
    return valid_loss, accuracy


def save_checkpoint(model, save_dir, optimizer, criterion, epochs):
    print("Saving checkpoint on : {}".format(save_dir))
    global training_data
    
    model.to('cpu')
    model.class_to_idx = training_data.class_to_idx
    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict,
                  'criterion': criterion,
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, save_dir)

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    model = checkpoint["model"]
    model.load_state_dict(checkpoint['state_dict'])
   
    return model

def process_image(image):
    
    image = Image.open(image)
    
    #resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    width = image.width
    height = image.height
    size = 256, 256
    
    if width > height:
        ratio = float(width) / float(height)
        newheight = ratio * size[0]
        image = image.resize((size[0], int(floor(newheight))), Image.ANTIALIAS)
    else:
        ratio = float(height) / float(width)
        newwidth = ratio * size[0]
        image = image.resize((int(floor(newwidth)), size[0]), Image.ANTIALIAS)

    transform = transforms.Compose([transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.299, 0.224, 0.225])])
    
    image_transformed = transform(image)
        
    return image_transformed


def predict(image_path, model, gpu, topk=5):

    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        
        image = process_image(image_path)
        image = torch.from_numpy(image.numpy()).type(torch.FloatTensor)
        image = image.unsqueeze_(0)
        image = image.to(device)
    
        output = model.forward(image)
    
        probabilities = torch.exp(output)
        probabilites_topk = probabilities.topk(topk)[0]
        probabilities_topk_indexes = probabilities.topk(topk)[1]

        probabilites_topk_list = np.array(probabilites_topk)[0]
        probabilities_topk_indexes_list = np.array(probabilities_topk_indexes[0])

        #Invert the dictionary
        class_to_idx = model.class_to_idx
        class_to_idx_items = {x: y for y, x in class_to_idx.items()}

        classes_topk_list = []
        for i in probabilities_topk_indexes_list:
            classes_topk_list += [class_to_idx_items[i]]
    return probabilites_topk_list, classes_topk_list

def show_results(image_path, probs, classes, category_names_input):

    with open(category_names_input, 'r') as f:
        category_names = json.load(f)
    
    image = Image.open(image_path)
    
    result_names = []
    
    for c in classes:
        result_names += [category_names[str(c)]]
    
    print("Top Results:")
    for i in range(len(result_names)):
        prob =  100 * probs[i]
        print(result_names[i]+ " - %.2f %%" % (prob))
        #print(result_names[i]+ " {}".format()  +  )  
