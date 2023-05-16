import os
import csv
import numpy as np
import sys
from PIL import Image

class DataBuilder:
    def __init__(self, image_dir, label_file, output_dir,output_file, output_label_file, target_size):
        self.image_dir = image_dir
        self.label_file = label_file
        self.target_size = target_size
        self.filtered_labels = []
        self.output_file = output_file
        self.output_label_file = output_label_file
        self.output_dir = output_dir
        self.labels = []

    def filter_labels(self, labels_to_ignore):
        # Load the labels from the CSV file and filter out the labels to ignore
        with open(self.label_file, 'r') as file:
            reader = csv.reader(file)
            labels = [row for row in reader]

        self.filtered_labels = [label for label in labels if label[2] not in labels_to_ignore]

    
    def process_images(self):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Open the output CSV file
        with open(self.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Iterate through the images in the directory
            for image_file in os.listdir(self.image_dir):
                if image_file.endswith('.jpg') or image_file.endswith('.png'):
                    image_path = os.path.join(self.image_dir, image_file)
                    label = self.get_label(image_file)
                    if label != None:
                        self.labels.append(label)
                        # Load, resize, and convert the image to an array of target size and three channels
                        image = Image.open(image_path)
                        image = image.resize(self.target_size)
                        image_array = np.array(image.convert('RGB'))

                        # Flatten the image array and append the label
                        flattened_image = image_array.flatten()
                        #row = np.concatenate(([label], flattened_image))

                        # Write the row to the CSV file
                        writer.writerow(flattened_image)
        self.store_labels()

    def get_index(self, image_file):
        for i in range(len(self.filtered_labels)):
            if image_file in self.filtered_labels[i][1]:
                return i
        return None


    def get_label(self, image_file):
        # Extract the label from the image file name or file itself
        # and return the corresponding label
        #We should search for labels ("sad" or "happy") appearing in the image file path (exact match not needed)
        #If one of the labels is found, and the third column matches it (i.e. "sad or happy"), then we return the label (0 or 1)
        #If no label is found, we return None, which means the image should be ignored
        ind = self.get_index(image_file)
        if ind is None:
            print(f"Image {image_file} not found in labels")
            return None
        if 'sad' in self.filtered_labels[ind][1] and self.filtered_labels[ind][2] == 'sad':
            return 0
        elif 'happy' in self.filtered_labels[ind][1] and self.filtered_labels[ind][2] == 'happy':
            return 1
        else:
            print(f"Image {image_file} not found in labels")
            return None
            

    def store_labels(self):
        # Store the label in a separate file (e.g., CSV or text file)
        # Implement your own logic here based on your desired output format
        with open(self.output_label_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for label in self.labels:
                writer.writerow(str(label))
            
        
# Example usage
builder = DataBuilder(
    image_dir='./images',
    label_file='./labels/labels.csv',
    output_dir='./output',
    output_file='./output/output.csv',
    output_label_file='./output/output_labels.csv',
    target_size=(64, 64)  # Example target size, adjust according to your needs
)

# Filter out labels to ignore
labels_to_ignore = ['surprise', 'anger', 'disgust', 'contempt', 'fear', 'neutral']  # Example labels to ignore
builder.filter_labels(labels_to_ignore)
#print(builder.filtered_labels)
#sys.exit()
# Process the images
builder.process_images()
