import torch
import numpy as np

class Predator:
	def __init__(self, x, y, speed_1=5, speed_0=2):
		"""
		arr[Comms]: You get some form of communication from your people
		x:
		y:
		floor_density:
		x_prey:
		y_prey:
		"""
		self.x = x
		self.y = y
		self.speed_0 = speed_0
		self.speed_1 = speed_1

	def record_action(self, state, action):
		print("Recorded! Predator")

	def scream(self, state):
		return state

	def take_action(self, state):
		return 1

class Prey:
	def __init__(self, x, y, speed_1=6, speed_0=4):
		self.x = x
		self.y = y
		self.speed_0 = speed_0
		self.speed_1 = speed_1

	def record_action(self, state, action):
		print("Recorded! Prey")

	def take_action(self, state):
		return 2