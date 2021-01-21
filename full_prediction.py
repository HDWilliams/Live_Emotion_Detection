import torch
import torch.nn.functional as F
from torchvision import transforms
from emotion_cnn import CNN
import numpy as np
from torch_utils import predict_emotion
import cv2


# load cascade module
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def get_full_prediction(img_file):
    """takes in img_file as string, outputs a prediction if a face is detected"""

    #read image
    img = np.fromfile(img_file, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    #turn image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    if len(faces) == 0:
        return img, 'No faces detected...'
    for (x, y, w, h) in faces:
        
        crop_img = gray[y:y+h, x:x+w]
        #detect emotion and softmax the output
        label, prob = predict_emotion(crop_img)
        full_label = str(label) + str(np.around(prob.item(), 3))
        #label with emotion and probability
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, full_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return img, 'Faces detected...'
   