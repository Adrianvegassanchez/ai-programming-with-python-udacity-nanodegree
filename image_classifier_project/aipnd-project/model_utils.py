from torchvision import models


def validate_model(arch):
    
    is_valid = False
    
    if arch == 'vgg16':
        is_valid = True
        model = models.vgg16(pretrained=True)
        features = list(model.classifier.children())[:0]
        input_dimension = model.classifier[len(features)].in_features
        print("Model : vgg16")
        
    elif arch == 'alexnet':
        is_valid = True
        model = models.alexnet(pretrained=True)
        features = list(model.classifier.children())[:1] 
        input_dimension = model.classifier[len(features)].in_features
        print("Model : Alexnet")
        
    else:
        model = models.vgg16(pretrained=True)
        features = list(model.classifier.children())[:0]
        input_dimension = model.classifier[len(features)].in_features
        print("Error Invalid model : {}".format(arch))
        print("Available models are vgg16 and alexnet")
        print("**A default vgg16 model has been created**")
    
    return is_valid, model, input_dimension