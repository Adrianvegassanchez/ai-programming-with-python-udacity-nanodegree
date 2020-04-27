import  argparse

from functions import load_checkpoint, predict, show_results

parser = argparse.ArgumentParser(description='Create Your Own Image Classifier - Command Line Application - Predict')

#Image
parser.add_argument('image_path', action='store', default='./flowers/test/78/image_01856.jpg', help = 'Enter an image path', type=str)
#Checkpoint
parser.add_argument('checkpoint', action='store', default='checkpoint.pth', help = 'Enter a checkpoint to load', type=str)
#Choose GPU use
parser.add_argument('--gpu', dest='gpu', action='store_true', default=False, help = 'Turn on the use of GPU')
#topk
parser.add_argument('--top_k', dest='top_k', action='store', default=5, help = 'Return top K predictions', type=int)
#category_names
parser.add_argument('--category_names', dest='category_names', action='store', default="cat_to_name.json", help = 'Category names to match with model', type=str)

arguments = parser.parse_args()

image_path = arguments.image_path
checkpoint = arguments.checkpoint
top_k = arguments.top_k
gpu = arguments.gpu
category_names = arguments.category_names

#Load checkpoint
print("Loading model from : {}\n".format(checkpoint))
model = load_checkpoint(checkpoint)
#Predict
probs, classes = predict(image_path, model, gpu, top_k)
#Show results
show_results(image_path, probs, classes, category_names)
