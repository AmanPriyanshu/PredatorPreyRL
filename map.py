import numpy as np
from perlin_noise import PerlinNoise
import pickle
import os
import cv2

class Landscape:
	def __init__(self, h=400, w=600, seed=None, var_ranges=[4, 20]):
		self.h = h
		self.w = w
		self.basemap = []
		self.seed = seed
		if self.seed is not None:
			if os.path.exists("./maps/map_seeded_"+str(self.seed)+".pkl"):
				with open("./maps/map_seeded_"+str(self.seed)+".pkl", "rb") as f:
					self.basemap = pickle.load(f)
			else:
				self.noise1 = PerlinNoise(octaves=3, seed=self.seed)
				self.noise2 = PerlinNoise(octaves=6, seed=self.seed)
				self.noise3 = PerlinNoise(octaves=12, seed=self.seed)
				for i in range(self.h):
					row = []
					for j in range(self.w):
						arr = [i/self.h, j/self.w]
						noise_val = self.noise1(arr)
						noise_val += 0.5 * self.noise2(arr)
						noise_val += 0.25 * self.noise3(arr)
						row.append(noise_val)
					self.basemap.append(row)
				self.basemap = np.array(self.basemap)
				self.basemap = (self.basemap - np.min(self.basemap))/(np.max(self.basemap) - np.min(self.basemap))
				self.basemap[self.basemap<0.5] = np.power(self.basemap[self.basemap<0.5], 2)

				if self.seed is not None:
					with open("./maps/map_seeded_"+str(self.seed)+".pkl", "wb") as f:
						pickle.dump(self.basemap, f)
		else:
			self.noise1 = PerlinNoise(octaves=3)
			self.noise2 = PerlinNoise(octaves=6)
			self.noise3 = PerlinNoise(octaves=12)
			for i in range(self.h):
				row = []
				for j in range(self.w):
					arr = [i/self.h, j/self.w]
					noise_val = self.noise1(arr)
					noise_val += 0.5 * self.noise2(arr)
					noise_val += 0.25 * self.noise3(arr)
					row.append(noise_val)
				self.basemap.append(row)
			self.basemap = np.array(self.basemap)
			self.basemap = (self.basemap - np.min(self.basemap))/(np.max(self.basemap) - np.min(self.basemap))
			self.basemap[self.basemap<0.5] = np.power(self.basemap[self.basemap<0.5], 2)

	def get_map(self):
		return self.basemap

	def get_truemap(self):
		self.truemap = np.zeros((self.h, self.w, 3))
		self.truemap[:, :, 0] = self.basemap
		self.truemap[:, :, 1] = (1-self.basemap)
		return self.truemap

if __name__ == '__main__':
	landscape = Landscape(seed=1)
	landscape.get_truemap()
	landscape.truemap[100:115, 100:115, :] = 0
	cv2.imshow("", landscape.truemap)
	cv2.waitKey(0)
	cv2.destroyAllWindows()