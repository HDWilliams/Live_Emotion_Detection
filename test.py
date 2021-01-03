import torch
import torch.nn.functional as F
from torchvision import transforms
from emotion_cnn import CNN
import numpy as np

import cv2


#initiaize and load module
model = CNN()
model.load_state_dict(torch.load('model0.6804249390456287.pth', map_location=torch.device('cpu')))
labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

#create input image transforms

transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((48, 48)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485], std=[0.229])])

#harrcasade classfier for detecting and cropping face
#before sending to emotion detection CNN

# load cascade module
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    #model trained on grayscale images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        
        crop_img = gray[y:y+h, x:x+w]
        #detect emotion and softmax the output
        crop_img = transform(crop_img)
        output = model(torch.unsqueeze(crop_img, 0))
        output = F.softmax(output)
        prob, pred = torch.max(output.data, dim=1)

        label = labels[pred.item()]
        full_label = str(label) + str(np.around(prob.item(), 3))

        #label with emotion and probability
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, full_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()