import torch
import torch.nn.functional as F
from torchvision import transforms
from emotion_cnn import CNN
import numpy as np

#initiaize and load module
model = CNN()
model.load_state_dict(torch.load('model0.6804249390456287.pth', map_location=torch.device('cpu')))
labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

#create input image transforms

transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize((48, 48)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485], std=[0.229])])


def predict(crop_img):
    #prediction function for precropped face image
    #outputs label as a string and probability of given label
    
    crop_img = transform(crop_img)
    output = model(torch.unsqueeze(crop_img, 0))
    output = F.softmax(output)
    prob, pred = torch.max(output.data, dim=1)

    label = labels[pred.item()]
    
    return label, prob
