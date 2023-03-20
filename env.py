from map import Landscape
from agents import Predator, Prey, PredatorBuffer, PredatorModel
import numpy as np
import cv2
import torch
from tqdm import trange

class Environment:
	def __init__(self, num_preds=5, seed=1):
		self.seed = seed
		self.landscape = Landscape(seed=self.seed)
		self.basemap = self.landscape.get_map()
		self.num_preds = num_preds
		self.buffer_holder = PredatorBuffer()
		self.predator_model = PredatorModel()
		self.predator_optimizer = torch.optim.Adam(self.predator_model.parameters(), lr=0.05) 
		self.game_condition = False
		self.criterion = torch.nn.MSELoss()
		self.step_count = 0
		
	def start(self, margin=50):
		x_pack, y_pack = np.random.randint(margin, self.basemap.shape[0]-margin), np.random.randint(margin, self.basemap.shape[1]-margin)
		xs = np.random.randint(-margin, margin, self.num_preds)+x_pack
		ys = np.random.randint(-margin, margin, self.num_preds)+y_pack
		self.preds = []
		for x,y in zip(xs, ys):
			self.preds.append(Predator(x, y, self.buffer_holder, self.predator_model))
		d = 0
		while d<20:
			prey_x, prey_y = np.random.randint(margin, self.basemap.shape[0]-margin), np.random.randint(margin, self.basemap.shape[1]-margin)
			d_x = np.power(xs-prey_x, 2)
			d_y = np.power(ys-prey_y, 2)
			d_arr = np.sqrt(d_x + d_y)
			d = np.min(d_arr)
		self.prey = Prey(prey_x, prey_y)

	def take_actions(self):
		prey_x, prey_y = self.prey.x, self.prey.y
		min_pred_x, min_pred_y = self.preds[0].x, self.preds[0].y
		min_d = np.sqrt((prey_x - self.preds[0].x)**2 + (prey_y - self.preds[0].y)**2)/10
		for pred in self.preds[1:]:
			d = np.sqrt((prey_x - pred.x)**2 + (prey_y - pred.y)**2)/10
			if d<min_d:
				d = min_d
				min_pred_x = pred.x
				min_pred_y = pred.y
		score, prey_action = self.prey.take_action([self.prey.x, self.prey.y, min_pred_x-self.prey.x, min_pred_y-self.prey.y, self.basemap[self.prey.x, self.prey.y]])
		prey_state = [self.prey.x, self.prey.y, min_pred_x-self.prey.x, min_pred_y-self.prey.y, self.basemap[self.prey.x, self.prey.y]]
		communications = []
		for pred in self.preds:
			scream = pred.scream([pred.x, pred.y, self.prey.x-pred.x, self.prey.y-pred.y, self.basemap[pred.x, pred.y]])
			communications.append(scream)
		communications = np.array(communications)
		communications = np.mean(communications, axis=0)
		pred_actions = []
		pred_states = []
		for pred in self.preds:
			score, pred_action = pred.take_action([pred.x, pred.y, self.prey.x-pred.x, self.prey.y-pred.y, self.basemap[pred.x, pred.y]]+[c for c in communications])
			pred_states.append([pred.x, pred.y, self.prey.x-pred.x, self.prey.y-pred.y, self.basemap[pred.x, pred.y]]+[c for c in communications])
			pred_actions.append(pred_action)
		prey_reward, pred_rewards = self.make_move(prey_state, prey_action, pred_states, pred_actions)
		self.step_count += 1
		return prey_reward, pred_rewards

	def make_move(self, prey_state, prey_action, pred_states, pred_actions, margin=10):
		density = self.basemap[self.prey.x, self.prey.y]
		if prey_action==0:
			self.prey.x = round(self.prey.x - (density*self.prey.speed_1 + (1-density)*self.prey.speed_0))
		if prey_action==1:
			self.prey.x = round(self.prey.x + (density*self.prey.speed_1 + (1-density)*self.prey.speed_0))
		if prey_action==2:
			self.prey.y = round(self.prey.y - (density*self.prey.speed_1 + (1-density)*self.prey.speed_0))
		if prey_action==3:
			self.prey.y = round(self.prey.y + (density*self.prey.speed_1 + (1-density)*self.prey.speed_0))
		
		if self.prey.x<margin:
			self.prey.x = margin
		if self.prey.x>self.basemap.shape[0]-1-margin:
			self.prey.x = self.basemap.shape[0]-1-margin
		if self.prey.y<margin:
			self.prey.y = margin
		if self.prey.y>self.basemap.shape[1]-1-margin:
			self.prey.y = self.basemap.shape[1]-1-margin

		for pred, pred_action in zip(self.preds, pred_actions):
			density = self.basemap[pred.x, pred.y]
			if pred_action==0:
				pred.x = round(pred.x - (density*pred.speed_1 + (1-density)*pred.speed_0))
			if pred_action==1:
				pred.x = round(pred.x + (density*pred.speed_1 + (1-density)*pred.speed_0))
			if pred_action==2:
				pred.y = round(pred.y - (density*pred.speed_1 + (1-density)*pred.speed_0))
			if pred_action==3:
				pred.y = round(pred.y + (density*pred.speed_1 + (1-density)*pred.speed_0))

			if pred.x<margin:
				pred.x = margin
			if pred.x>self.basemap.shape[0]-margin:
				pred.x = self.basemap.shape[0]-margin
			if pred.y<margin:
				pred.y = margin
			if pred.y>self.basemap.shape[1]-margin:
				pred.y = self.basemap.shape[1]-margin
		prey_reward, pred_rewards = self.get_rewards()
		communications = []
		for pred in self.preds:
			scream = pred.scream([pred.x, pred.y, self.prey.x-pred.x, self.prey.y-pred.y, self.basemap[pred.x, pred.y]])
			communications.append(scream)
		communications = np.array(communications)
		communications = np.mean(communications, axis=0)
		next_states = []
		for pred in self.preds:
			next_states.append([pred.x, pred.y, self.prey.x-pred.x, self.prey.y-pred.y, self.basemap[pred.x, pred.y]]+[c for c in communications])
		prey_x, prey_y = self.prey.x, self.prey.y
		min_pred_x, min_pred_y = self.preds[0].x, self.preds[0].y
		min_d = np.sqrt((prey_x - self.preds[0].x)**2 + (prey_y - self.preds[0].y)**2)/10
		for pred in self.preds[1:]:
			d = np.sqrt((prey_x - pred.x)**2 + (prey_y - pred.y)**2)/10
			if d<min_d:
				d = min_d
				min_pred_x = pred.x
				min_pred_y = pred.y
		self.prey.record(prey_state, prey_action, prey_reward, [self.prey.x, self.prey.y, min_pred_x-self.prey.x, min_pred_y-self.prey.y, self.basemap[self.prey.x, self.prey.y]])
		for pred, pred_state, pred_action, pred_reward, next_state in zip(self.preds, pred_states, pred_actions, pred_rewards, next_states):
			pred.record(pred_state, pred_action, pred_reward, next_state)
		return prey_reward, pred_rewards

	def get_rewards(self):
		prey_x, prey_y = self.prey.x, self.prey.y
		d_arr = []
		for pred in self.preds:
			d = np.sqrt((prey_x - pred.x)**2 + (prey_y - pred.y)**2)/10
			d_arr.append(d)
		predator_rewards = [72.2-d for d in d_arr]
		prey_reward = min(d_arr)
		if prey_reward<5 or self.step_count>500:
			prey_reward = -100
			predator_rewards = [100 for _ in range(len(predator_rewards))]
			self.game_condition = True
			self.step_count = 0
		return prey_reward, predator_rewards

	def plot(self, visualize=True):
		colored_map = self.landscape.get_truemap()
		colored_map[self.prey.x-5:self.prey.x+5, self.prey.y-5:self.prey.y+5, :] = 0
		for pred in self.preds:
			colored_map[pred.x-7:pred.x+7, pred.y-7:pred.y+7, :] = 1
		#return colored_map
		if visualize:
			cv2.imshow("", colored_map)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		else:
			return colored_map

	def train_step(self, batch_size=8, epochs=100):
		for epoch in range(epochs):
			train_prey_indices = np.random.randint(0, len(self.prey.buffer), batch_size)
			outs = []
			for index in train_prey_indices:
				prey_data = self.prey.buffer[index]
				with torch.no_grad():
					out = self.prey.model(torch.tensor(prey_data["next_state"]).float()).numpy()
					outs.append(np.max(out))
			expected = torch.tensor(np.array(outs)).float()
			for _ in range(epochs):
				self.prey.model.train()
				out = self.prey.model(torch.tensor([self.prey.buffer[i]["state"] for i in train_prey_indices]).float())
				actions = torch.tensor([self.prey.buffer[i]["action"] for i in train_prey_indices])
				prediction = out.gather(1, actions.unsqueeze(1))
				rewards = torch.tensor([self.prey.buffer[i]["reward"] for i in train_prey_indices]).float()
				target = rewards+0.9*expected
				target = target.unsqueeze(1)
				self.prey.optimizer.zero_grad()
				loss = self.criterion(prediction, target)
				loss.backward()
				self.prey.optimizer.step()
				loss_prey = loss.detach().item()

			train_pred_indices = np.random.randint(0, len(self.buffer_holder.buffer), batch_size)
			outs = []
			for index in train_pred_indices:
				pred_data = self.buffer_holder.buffer[index]
				with torch.no_grad():
					out, _ = self.predator_model(torch.tensor(pred_data["next_state"]).float())
					out = out.numpy()
					outs.append(np.max(out))
			expected = torch.tensor(np.array(outs)).float()
			for _ in range(epochs):
				self.predator_model.train()
				out, _ = self.predator_model(torch.tensor([self.buffer_holder.buffer[i]["state"] for i in train_pred_indices]).float())
				actions = torch.tensor([self.buffer_holder.buffer[i]["action"] for i in train_pred_indices])
				prediction = out.gather(1, actions.unsqueeze(1))
				rewards = torch.tensor([self.buffer_holder.buffer[i]["reward"] for i in train_pred_indices]).float()
				target = rewards+0.9*expected
				target = target.unsqueeze(1)
				self.predator_optimizer.zero_grad()
				loss = self.criterion(prediction, target)
				loss.backward()
				loss_preds = loss.detach().item()
				self.predator_optimizer.step()
		return loss_prey, loss_preds

	def reset(self, margin=50):
		x_pack, y_pack = np.random.randint(margin, self.basemap.shape[0]-margin), np.random.randint(margin, self.basemap.shape[1]-margin)
		xs = np.random.randint(-margin, margin, self.num_preds)+x_pack
		ys = np.random.randint(-margin, margin, self.num_preds)+y_pack
		for pred, x,y in zip(self.preds, xs, ys):
			pred.x = x
			pred.y = y
			pred.epsilon = 0.0
		d = 0
		while d<20:
			prey_x, prey_y = np.random.randint(margin, self.basemap.shape[0]-margin), np.random.randint(margin, self.basemap.shape[1]-margin)
			d_x = np.power(xs-prey_x, 2)
			d_y = np.power(ys-prey_y, 2)
			d_arr = np.sqrt(d_x + d_y)
			d = np.min(d_arr)
		self.prey.x = prey_x
		self.prey.y = prey_y
		self.prey.epsilon = 0.0
		self.game_condition = False

if __name__ == '__main__':
	env = Environment()
	env.start()
	N = 1000
	bar = trange(N)
	loss_prey = ""
	loss_preds = ""
	for i in bar:
		if env.game_condition:
			env.reset()
		prey_reward, pred_rewards = env.take_actions()
		if i%8==0 and i>1:
			loss_prey, loss_preds = env.train_step()
			loss_prey = str(round(loss_prey, 3))
			loss_preds = str(round(loss_preds, 3))
			print()
		bar.set_description(str({"prey_reward": round(prey_reward, 3), "pred_rewards": round(np.mean(pred_rewards), 3), "loss_prey": loss_prey, "loss_preds": loss_preds}))
		bar.update()
		if i>N:
			break
	bar.close()
	torch.save(env.prey.model.state_dict(), "prey_model.pt")
	torch.save(env.predator_model.state_dict(), "pred_model.pt")

	env.reset()
	for i in range(500):
		env.prey.epsilon = 1.0
		for pred in env.preds:
			pred.epsilon = 1.0
		env.take_actions()
		env.plot()