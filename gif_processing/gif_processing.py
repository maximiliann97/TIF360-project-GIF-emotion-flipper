import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import os
import sys
import cv2
import numpy as np
from PIL import ImageOps

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Class to view a GIF
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
		
# Class to modify a GIF
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

	# Resets to empty GIF
	def reset(self):
		self.frames = []
		self.n_frames = 0
		self.frame_durations = []
		self.faces = []
		self.coordinates = []
		self.flipped_frames = []

	# Loads a GIF from a path
	def load_frames(self, filename):
		# Resets all variables
		self.reset()

		# Create a default root window
		temp_root = tk.Tk()
		temp_root.withdraw()

		# Loads gif file
		gif = Image.open("data/" + filename)

		# Extracts total number of frames
		self.n_frames = gif.n_frames

		# Extracts frame durations
		self.frame_durations = [gif.info['duration'] for i in range(self.n_frames)]

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
		for i in range(len(self.frames)):
			faces_cut_out, coordinates = self.list_of_faces(self.frames[i])
			self.faces[i] = (faces_cut_out)
			self.coordinates.append(coordinates)


	def flip_faces(self, margin=10, fade_type="linear", sig_param=4):
		for i in range(self.n_frames):
			flipped_frame = self.frames[i].copy()
			for j in range(len(self.faces[i])):
				face = self.faces[i][j]
				face_img = Image.fromarray(face)

				# Resizes face to 96x96 pixels
				face_img = face_img.resize((96, 96))
				flipped_face = ImageOps.flip(face_img)
				flipped_face = flipped_face.resize((face.shape[1], face.shape[0]))
				
				# Create an alpha mask
				width, height = flipped_face.size
				mask = Image.new('L', (width, height), 0)

				# Create a mask for the fade effect
				fade_mask = Image.new('L', (width, height), 255)


				for y in range(height):
					for x in range(width):
						# Calculate distance to nearest edge
						dist_to_edge = min(x, width-x, y, height-y)
						# If the distance is less than the margin, decrease the opacity linearly
						if fade_type == "linear":
							opacity = int(255 * (dist_to_edge / margin))
						elif fade_type == "sigmoid":
							opacity = int(255 * sigmoid(2*sig_param * (dist_to_edge / margin) - sig_param))
						
						fade_mask.putpixel((x, y), opacity)

				# Blend the two masks
				mask = Image.blend(mask, fade_mask, alpha=1)

				# Create a 4-channel image (RGB + alpha)
				flipped_face.putalpha(mask)

				# Paste the flipped face onto the frame
				flipped_frame.paste(flipped_face, tuple(self.coordinates[i][j][0][0:2]), flipped_face)
			self.flipped_frames.append(flipped_frame)


	def build_flipped_gif(self, filename):
		#Build GIF from flipped_frames and save to path
		#This function assumes that the frames have been loaded
		#and that the faces have been detected and flipped
		#and that the flipped frames have been built
		#and that the frame durations have been loaded
		flipped_gif = Image.new('RGB', self.flipped_frames[0].size)
		flipped_gif.save("output/" + filename + ".gif", save_all=True, append_images=self.flipped_frames[:-1], duration=self.frame_durations, loop=0)
		
		
if __name__ == "__main__":	
	# Script path
	#Initiate the flipper
	gif_flipper = GifFlipper()

	#View the original gif once
	#GifViewer(gif_path).view_gif()

	gif_flipper.load_frames("mike.gif")
	gif_flipper.detect_faces()
	gif_flipper.flip_faces(margin=35, fade_type="sigmoid")
	gif_flipper.build_flipped_gif("testing2")
	#View the flipped gif once
	#GifViewer(flipped_gif_path).view_gif()



