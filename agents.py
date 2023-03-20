import torch
import numpy as np

class PredatorBuffer:
	def __init__(self):
		self.buffer = []
		self.buffer_max = 1000

	def record(self, state, action, reward, next_state):
		self.buffer.append({"state": state, "action": action, "reward": reward, "next_state": next_state})
		if len(self.buffer)>self.buffer_max:
			self.buffer = self.buffer[1:]

class PredatorModel(torch.nn.Module):
	def __init__(self):
		super(PredatorModel, self).__init__()
		self.linear1 = torch.nn.Linear(13, 16)
		self.activation = torch.nn.ReLU()
		self.linear2 = torch.nn.Linear(16, 8)
		self.linear3 = torch.nn.Linear(8, 4)

	def forward(self, state):
		x = self.linear1(state)
		x = self.activation(x)
		x = self.linear2(x)
		x_ = self.activation(x)
		x = self.linear3(x_)
		return x, x_

class PreyModel(torch.nn.Module):
	def __init__(self):
		super(PreyModel, self).__init__()
		self.linear1 = torch.nn.Linear(5, 16)
		self.activation = torch.nn.ReLU()
		self.linear2 = torch.nn.Linear(16, 8)
		self.linear3 = torch.nn.Linear(8, 4)

	def forward(self, state):
		x = self.linear1(state)
		x = self.activation(x)
		x = self.linear2(x)
		x_ = self.activation(x)
		x = self.linear3(x_)
		return x


class Predator:
	def __init__(self, x, y, buffer_holder, model, speed_1=2, speed_0=5):
		self.x = x
		self.y = y
		self.speed_0 = speed_0
		self.speed_1 = speed_1
		self.buffer_holder = buffer_holder
		self.model = model
		self.epsilon = 0.0
		self.epsilon_step = 0.02

	def record(self, state, action, reward, next_state):
		self.buffer_holder.record(state, action, reward, next_state)

	def scream(self, state):
		state = state + [0]*8
		state = torch.tensor(state).float()
		with torch.no_grad():
			_, scream_vec = self.model(state)
		return scream_vec.numpy()

	def take_model_action(self, state):
		state = torch.tensor(state).float()
		with torch.no_grad():
			y, _ = self.model(state)
		return y.numpy()

	def take_action(self, state, only_model=False):
		self.epsilon += self.epsilon_step
		y = self.take_model_action(state)
		if only_model:
			index = np.argmax(y)
		else:
			if np.random.random()>self.epsilon:
				index = np.random.randint(0, 4) 
			else:
				index = np.argmax(y)
		return y[index], index

class Prey:
	def __init__(self, x, y, speed_1=4, speed_0=6):
		self.x = x
		self.y = y
		self.speed_0 = speed_0
		self.speed_1 = speed_1
		self.buffer = []
		self.buffer_max = 1000
		self.model = PreyModel()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05) 
		self.epsilon = 0.0
		self.epsilon_step = 0.02

	def record(self, state, action, reward, next_state):
		self.buffer.append({"state": state, "action": action, "reward": reward, "next_state": next_state})
		if len(self.buffer)>self.buffer_max:
			self.buffer = self.buffer[1:]

	def take_model_action(self, state):
		state = torch.tensor(state).float()
		with torch.no_grad():
			y = self.model(state)
		return y.numpy()

	def take_action(self, state, only_model=False):
		self.epsilon += self.epsilon_step
		y = self.take_model_action(state)
		if only_model:
			index = np.argmax(y)
		else:
			if np.random.random()>self.epsilon:
				index = np.random.randint(0, 4) 
			else:
				index = np.argmax(y)
		return y[index], index