import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import os

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
				frame = ImageTk.PhotoImage(gif)
				self.frames.append(frame)
				self.frame_durations.append(gif.info['duration'])
				gif.seek(gif.tell() + 1)
		except EOFError:
			pass

	def update_gif(self):
		self.label.config(image=self.frames[self.current_frame])
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

		# List of all faces in a frame as Image objects, for each frame
		self.faces = []

	def reset(self):
		self.frames = []
		self.n_frames = 0
		self.frame_durations = []
		self.faces = []

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


if __name__ == "__main__":

	# Script path
	script_path = os.path.dirname(os.path.realpath(__file__))

	gif_flipper = GifFlipper()
	gif_path = os.path.join(script_path, "data/mike.gif")
	gif_flipper.load_frames(gif_path)

	GifViewer(gif_path).view_gif()
