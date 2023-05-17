import cv2
import os
import glob

# Path to the input images
input_folder_path = 'images'

# Path to the output folder
output_folder_path = 'faces'

# Ensure output directory exists
#if not os.path.exists(output_folder_path):
#    os.makedirs(output_folder_path)

# Load Haarcascade face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Image counter
img_counter = 0

# Iterate over each image in the folder
for img_file in glob.glob(input_folder_path + ".jpg"):
    # Load image
    img = cv2.imread(img_file)
    
    print("Ah")

    # Convert color style from BGR to gray (necessary for face detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Extract each face
    for (x, y, w, h) in faces:
        # Create rectangle around the face (for testing)
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract face
        face_img = img[y:y+h, x:x+w]
        
        # Save face
        face_file = output_folder_path + 'face_' + str(img_counter) + '.jpg'
        cv2.imwrite(face_file, face_img)
        
        img_counter += 1

print('Done. Detected faces have been saved to: ' + output_folder_path)
