from deepface import DeepFace as df
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
#Unfortunately, this cannot be a method of the class :()
def analyze_emotions(img_path):
        #This function takes in an image path and returns a list of dictionaries with the emotions
        emo_list = df.analyze(img_path, actions=['emotion'], enforce_detection=False)
        return emo_list


class ExpressionEvaluator:
    def __init__(self, real_images_path, fake_images_path):
        self.real_images_path = real_images_path#directory path
        self.fake_images_path = fake_images_path#directory path
        self.real_predictions = []
        self.real_confidences = []
        self.fake_predictions = []
        self.fake_confidences = []

    def get_prediction(self, emo_list):
        #This function takes in a DeepFace.analyze() output and returns the predicted emotion
        emotions = emo_list[0]['emotion']
        prediction = max(emotions, key=emotions.get)
        return prediction 
    def get_confidence(self, emo_list, prediction = None):
        #This function takes in a DeepFace.analyze() output and returns the confidence of the prediction
        emotions = emo_list[0]['emotion']
        if prediction == None:
            prediction = max(emotions, key=emotions.get)
        confidence = int(np.round(emotions[prediction]))
        return confidence
    
    def evaluate_performance(self):
        #This function loops through images in the real and fake directories and compares performance of altered images
        #Expeting a one to one correspondence between real and fake images
        for real_im, fake_im in zip(os.listdir(real_images_path),os.listdir(fake_images_path)):
            try:
                fake_path = os.path.join(fake_images_path, fake_im)
                real_path = os.path.join(real_images_path, real_im)
                fake_emo_list = analyze_emotions(fake_path)
                real_emo_list = analyze_emotions(real_path)
                fake_pred = self.get_prediction(fake_emo_list)
                fake_conf = self.get_confidence(fake_emo_list, fake_pred)
                real_pred = self.get_prediction(real_emo_list)
                real_conf = self.get_confidence(real_emo_list, real_pred)
                self.real_predictions.append(real_pred)
                self.real_confidences.append(real_conf)
                self.fake_predictions.append(fake_pred)
                self.fake_confidences.append(fake_conf)
            except:
                pass #face not detected
    def save_performance(self):
        #This function saves the performance of the model in a text file
        with open("fake_prediction.txt", "w") as f:
            f.write(str(self.fake_predictions))
        with open("fake_confidence.txt", "w") as f:
            f.write(str(self.fake_confidences))
        with open("real_prediction.txt", "w") as f:
            f.write(str(self.real_predictions))
        with open("real_confidence.txt", "w") as f:
            f.write(str(self.real_confidences))

    def display_analysis(self, fake_path, fake_prediction, fake_confidence, real_path = None, real_prediction = None, real_confidence = None):
        if real_path == None or real_prediction == None or real_confidence == None:
            print("displaying fake only")
            plt.figure(figsize=(3,5))
            plt.subplot(1, 1, 1)
            plt.title("Fake Image")
            plt.imshow(plt.imread(fake_path))
            plt.xlabel("Prediction: " + fake_prediction + "\nConfidence: " + str(fake_confidence) + "%")
            plt.xticks([])
            plt.yticks([])
            plt.show()
        else:
            print("displaying real and fake")
            plt.figure(figsize=(5,5))
            plt.subplot(1, 2, 1)
            plt.title("Real Image")
            plt.imshow(plt.imread(real_path))
            plt.xlabel("Prediction: " + real_prediction + "\n Confidence: " + str(real_confidence) + "%")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1, 2, 2)
            plt.title("Fake Image")
            plt.imshow(plt.imread(fake_path))
            plt.xlabel("Prediction: " + fake_prediction + "\nConfidence: " + str(fake_confidence) + "%")
            plt.xticks([])
            plt.yticks([])
            plt.show()



if __name__ == "__main__":
    demo = False
    if demo:
        my_eval = ExpressionEvaluator("real_images", "fake_images")
        this_path = os.path.dirname(os.path.abspath(__file__))
        fake_image = str(os.path.join(this_path, "test_images/fake_happy.png"))
        real_image = str(os.path.join(this_path, "test_images/orig_sad.png"))

        fake_emo_list = analyze_emotions(fake_image)
        real_emo_list = analyze_emotions(real_image)

        fake_prediction = my_eval.get_prediction(fake_emo_list)
        real_prediction = my_eval.get_prediction(real_emo_list)
        fake_confidence = my_eval.get_confidence(fake_emo_list, prediction=fake_prediction)
        real_confidence = my_eval.get_confidence(real_emo_list, prediction=real_prediction)
        my_eval.display_analysis(fake_image, fake_prediction, fake_confidence)
        my_eval.display_analysis(fake_image, fake_prediction, fake_confidence, real_image, real_prediction, real_confidence)
    this_path = os.path.dirname(os.path.abspath(__file__))
    fake_images_path = str(os.path.join(this_path, "fake_images/"))#does not exist yet
    real_images_path = str(os.path.join(this_path, "real_images/"))#does not exist yet
    my_eval = ExpressionEvaluator(real_images_path, fake_images_path)
    my_eval.evaluate_performance()
    print(my_eval.fake_confidences)
    my_eval.save_performance()