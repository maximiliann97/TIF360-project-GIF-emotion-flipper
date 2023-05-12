import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import os
import sys
import cv2
import numpy as np


class GifViewer:
	def __init__(self, gif_path, loop=True):
		self.root = tk.Tk()
		self.root.title("GIF Viewer")
		self.loop = loop
		self.current_frame = 0
		self.label = tk.Label(self.root)
		gif = Image.open(gif_path)
		self.frames = []
		self.frame_durations = []

		try:
			while True:
				frame = ImageTk.PhotoImage(gif, master = self.root)
				self.frames.append(frame)
				self.frame_durations.append(gif.info['duration'])
				gif.seek(gif.tell() + 1)
		except EOFError:
			pass

	def update_gif(self):
		self.label.config(image= (self.frames[self.current_frame]))
		self.current_frame = (self.current_frame + 1) % len(self.frames)

		delay = self.frame_durations[self.current_frame]
		if self.loop or self.current_frame != 0:
			self.root.after(delay, self.update_gif)

	def view_gif(self):
		self.label.pack()
		self.update_gif()
		self.root.mainloop()
		


class GifFlipper:
	def __init__(self):
		# List of PIL.Image.Image frames
		self.frames = []
		self.n_frames = 0

		# List of ms per frame (int)
		self.frame_durations = []

		# List of lists of all faces in a frame as Image objects, for each frame
		self.faces = []
		
		#List of coordinates of faces in each frame. Each element is a list of coordinates on the form [x,y,w,h]
		self.coordinates = []

	def reset(self):
		self.frames = []
		self.n_frames = 0
		self.frame_durations = []
		self.faces = []
		self.coordinates = []

	def load_frames(self, path):
		# Resets all variables
		self.reset()

		# Create a default root window
		temp_root = tk.Tk()
		temp_root.withdraw()

		# Loads gif file
		gif = Image.open(path)

		# Extracts total number of frames
		self.n_frames = gif.n_frames

		for i in range(self.n_frames):
			frame = ImageTk.PhotoImage(gif)

			# Appends Image object to frames
			self.frames.append(ImageTk.getimage(frame))

			# Adds empty list to faces list
			self.faces.append([])

			# Updates the pointer if it is not the last iteration
			if i < self.n_frames - 1:
				gif.seek(gif.tell() + 1)
	
	#This function takes an image as input and returns a list of faces (numpy arrays), along with their coordinates.
	#typically img will be a PIL Image object, but any image that can be cast to a numpy array will work
	def list_of_faces(self, img):
		face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		rgb_img = img.convert('RGB')
		# detect faces
		img_arr = np.array(rgb_img)
		faces = face_cascade.detectMultiScale(img_arr, 1.3, 5)
		# cut out faces, save them in a list
		faces_list = []
		coordinates_list = []
		for (x, y, w, h) in faces:
			faces_list.append(img_arr[y:y + h, x:x + w])
			coordinates_list.append([[x,y,w,h]])
		return faces_list, coordinates_list
	
	def detect_faces(self):
		#This function assumes that the frames have been loaded
		for frame in self.frames:
			faces_cut_out, coordinates = self.list_of_faces(frame)
			self.faces.append(faces_cut_out)
			self.coordinates.append(coordinates)


if __name__ == "__main__":

	# Script path
	script_path = os.path.dirname(os.path.realpath(__file__))
	#Initiate the flipper
	gif_flipper = GifFlipper()
	gif_path = os.path.join(script_path, "data/mike.gif")
	#View the original gif once
	GifViewer(gif_path).view_gif()

	gif_flipper.load_frames(gif_path)
	gif_flipper.detect_faces()
	print(f"the length of the faces list is {len(gif_flipper.faces)} and the last element looks like this: {gif_flipper.faces[-1]}")
	print(f"the length of the coordinates list is {len(gif_flipper.coordinates)} and the last element looks like this: {gif_flipper.coordinates[-1][-1]}")
	print(np.shape(gif_flipper.faces[-5][-1]))	
	print(np.shape(gif_flipper.coordinates[-1][-1]))

