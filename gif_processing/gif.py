import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageTk

cv2.ocl.setUseOpenCL(False)


def load_gif_frames(gif_path):
	gif = Image.open(gif_path)

	frames = []

	frame_durations = []

	try:
		while True:
			frame = ImageTk.PhotoImage(gif)
			frames.append(frame)
			frame_durations.append(gif.info['duration'])
			gif.seek(gif.tell() + 1)
	except EOFError:
		pass


	return frames, frame_durations


#This function takes an image as input and returns a list of faces, along with their coordinates
def list_of_faces(img):
	# convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# detect faces
	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	# cut out faces, save them in a list
	faces_list = []
	coordinates_list = []
	for (x, y, w, h) in faces:
		faces_list.append(img[y:y + h, x:x + w])
		coordinates_list.append([[x,y,w,h]])
	return faces_list, coordinates_list


#this function takes an image, a list of faces, and a list of coordinates as input and returns an image with the faces replaced by the faces in the list
def replace_faces(img, faces_list, coordinates_list):
    #coordinates are ordered as follows: x, y, w, h
    for i, (x,y,w,h) in zip(range(len(faces_list)), coordinates_list):
        img[y:y+h, x:x+w] = faces_list[i]
    return img




class GifViewer:
	def __init__(self, gif_path, loop=True):
		self.root = tk.Tk()
		self.root.title("GIF Viewer")
		self.loop = loop
		self.frames, self.frame_durations = load_gif_frames(gif_path)
		self.current_frame = 0
		self.label = tk.Label(self.root)
		print(type(self.frames))

	def view_gif(self):
		self.label.pack()
		self.update_gif()
		self.root.mainloop()

	def update_gif(self):
		self.label.config(image=self.frames[self.current_frame])
		self.current_frame = (self.current_frame + 1) % len(self.frames)

		delay = self.frame_durations[self.current_frame]
		if self.loop or self.current_frame != 0:
			self.root.after(delay, self.update_gif)

	def add_facial_detection(self):
		face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

		frames_copy = []

		for frame in self.frames:
			# Convert ImageTk.PhotoImage to PIL.Image
			image = ImageTk.getimage(frame)

			# Convert the PIL Image to OpenCV format (BGR)
			frame_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

			# Convert the frame to grayscale for face detection
			gray = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY)

			# Detect faces in the frame
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)

			# Draw rectangles around detected faces on the original PIL Image
			draw = ImageDraw.Draw(image)
			for (x, y, w, h) in faces:
				draw.rectangle((x, y, x + w, y + h), outline="red", width=2)

			# Convert the PIL Image back to ImageTk.PhotoImage
			frames_copy.append(ImageTk.PhotoImage(image))

		self.frames = frames_copy


if __name__ == "__main__":
	gif_view = GifViewer(r"data/mike.gif")
	gif_view.add_facial_detection()
	gif_view.view_gif()

