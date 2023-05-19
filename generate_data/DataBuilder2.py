import os
import csv
import numpy as np
import sys
from PIL import Image
import cv2

class FaceCropper:
    '''Given a path to a directory, FaceCropper will iterate through the files and crop the faces out of the images.
    These cropped faces will be resized to the size of the original image, then saved in a given output directory.'''
    def __init__(self, image_dir, output_dir, target_size):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.target_size = target_size
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
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
            #if image_file.endswith('.jpg') or image_file.endswith('.png'):
            image_path = os.path.join(self.image_dir, image_file)
            cropped_image = self.crop_face(image_path)
            self.store_cropped_face(cropped_image, image_file)

if __name__ == '__main__':
    # Create a FaceCropper object
    face_cropper_happy = FaceCropper(image_dir='./happy', output_dir='./cropped_happy_detected_only', target_size=(96, 96))

    # Crop the faces from the images
    face_cropper_happy.process_images()

    # Create a FaceCropper object
    face_cropper_sad = FaceCropper(image_dir='./sad', output_dir='./cropped_sad_detected_only', target_size=(96, 96))

    # Crop the faces from the images
    face_cropper_sad.process_images()
    
        