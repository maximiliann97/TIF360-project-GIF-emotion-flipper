import os
import csv
import numpy as np
import sys
from PIL import Image
import cv2
import torch
from labeling_data import ResNet
import torchvision

class FaceCropper:
    '''Given a path to a directory, FaceCropper will iterate through the files and crop the faces out of the images.
    These cropped faces will be resized to the size of the original image, then saved in a given output directory.'''
    def __init__(self, image_dir, output_dir, target_size, model, label):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.model = model
        self.label = label
    
    def crop_face(self, image_path):
        '''Given an image, this function will return a cropped version of the image containing only the face.'''
        # Load the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect face in the image. Assume there is only one face in the image.
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # Crop the faces and save them to the output directory
        for (x,y,w,h) in faces:
            cropped = img[y:y+h, x:x+w]
            cropped = cv2.resize(cropped, self.target_size)
            return cropped
    def store_cropped_face(self, cropped_image, file_name):
        if cropped_image is not None:
            cv2.imwrite(os.path.join(self.output_dir, file_name), cropped_image)
        else:
            pass#cv2.imwrite(os.path.join(self.output_dir, file_name), cv2.imread(os.path.join(self.image_dir, file_name)))
    
    def process_images(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Iterate through the images in the directory
        for image_file in os.listdir(self.image_dir):
            path = os.path.join(self.image_dir, image_file)
            img = Image.open(path)
            #Convert to grayscale
            img = img.convert('L')
            #Resize to 48x48
            img = img.resize((48,48))
            #Convert to tensor
            img = torchvision.transforms.ToTensor()(img)
            #Add batch dimension
            img = img.unsqueeze(0)
            #Predict
            pred = self.model(img)
            #Get the index of the highest probability
            #pred = int(torch.argmax(pred))
            if float(pred[0][self.label]) < -1.45 or int(torch.argmax(pred)) != self.label:
                #print("Made it here also")
                continue
            #print("Made it here")
            #if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(self.image_dir, image_file)
            cropped_image = self.crop_face(image_path)
            self.store_cropped_face(cropped_image, image_file)

if __name__ == '__main__':
    #Load model 
    #Load the model
    modelDict = torch.load('emotion_detection_model_state.pth', map_location=torch.device('cpu'))
    model = ResNet(1,5)
    model.load_state_dict(state_dict = modelDict)
    # Create a FaceCropper object
    #Labels 0-4: 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise'
    label = 3
    if label == 1:
        image_dir = './happy'
        output_dir = './approved_happy'
    elif label == 3:
        image_dir = './sad'
        output_dir = './approved_sad'
    face_cropper_happy = FaceCropper(image_dir=image_dir, output_dir= output_dir, target_size=(96, 96), model = model, label = label)

    # Crop the faces from the images
    face_cropper_happy.process_images()

    
        