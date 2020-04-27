import torch
from torchvision import datasets, transforms, models, utils


def make_train_transforms():
    return transforms.Compose([transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
def make_test_transforms():
    return transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

def make_validation_transforms():
    return transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
def load_dataset(image_folder, transform):
    return datasets.ImageFolder(image_folder, transform = transform)

# TODO: Using the image datasets and the trainforms, define the dataloaders
def make_dataloaders(image_dataset):
    return torch.utils.data.DataLoader(image_dataset, batch_size = 32, shuffle=True)

#Print some info about data :
def print_dataset_info(data):
    print("Dataset size: " + str(len(data)) + " and " + str(len(data.classes)) + " classes")



