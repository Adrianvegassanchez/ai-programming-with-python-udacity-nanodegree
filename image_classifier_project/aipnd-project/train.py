import  argparse
from functions import load_data, create_model, train, save_checkpoint

parser = argparse.ArgumentParser(description='Create Your Own Image Classifier - Command Line Application - Train')

#Data directory
parser.add_argument('data_dir', action= 'store', default='./flowers', help = 'Enter data directory')
#Save a checkpoint 
parser.add_argument('--save_dir', dest='save_dir', action='store', default='./checkpoint.pth', help = 'Choose where save a checkpoint')
#Choose arch ( vgg16, densenet121, alexnet )
parser.add_argument('--arch', dest='arch', action='store', default='vgg16', type = str, help = 'Availables : Vgg16 or Alexnet')
#Hyperparameters:
#lr
parser.add_argument('--learning_rate', dest= 'learning_rate', action='store', default=0.0008, help = 'Enter a learning rate value', type=float)
#hidden_units
parser.add_argument('--hidden_units', dest= 'hidden_units', action='store', default='500', type=int, help = 'Enter the number of hidden units')
#epochs
parser.add_argument('--epochs', dest= 'epochs', action='store', default='3', type=int, help = 'Enter the number of epochs')
#Choose GPU use
parser.add_argument('--gpu', dest='gpu', action='store_true', default=False, help = 'Turn on the use of GPU')

arguments = parser.parse_args()

data_dir = arguments.data_dir
save_dir = arguments.save_dir
arch = arguments.arch
learning_rate = arguments.learning_rate
hidden_units = arguments.hidden_units
epochs = arguments.epochs
gpu = arguments.gpu

print("Data Info :\n") 
train_dataloader, test_dataloader, validation_dataloader = load_data(data_dir)
print("\nModel Info :\n") 
model = create_model(arch, hidden_units)
print("\nTrain starts:\n")

if gpu:
    print("GPU activated")
else: 
    print("GPU is not activated")

print("Training epochs : {}".format(str(epochs)))
print("Learning rate : {:.4f}".format(learning_rate))
      
model, optimizer, criterion = train(model, train_dataloader, validation_dataloader, learning_rate, gpu, epochs, 40)
print("\nSave checkpoint:\n")
save_checkpoint(model, save_dir, optimizer, criterion, epochs)


