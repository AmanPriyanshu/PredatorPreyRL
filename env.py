from map import Landscape
from agents import Predator, Prey
import numpy as np
import cv2

class Environment:
	def __init__(self, num_preds=5, seed=1):
		self.seed = seed
		self.landscape = Landscape(seed=self.seed)
		self.basemap = self.landscape.get_map()
		self.num_preds = num_preds
		
	def start(self, margin=50):
		x_pack, y_pack = np.random.randint(margin, self.basemap.shape[0]-margin), np.random.randint(margin, self.basemap.shape[1]-margin)
		xs = np.random.randint(-margin, margin, self.num_preds)+x_pack
		ys = np.random.randint(-margin, margin, self.num_preds)+y_pack
		self.preds = []
		for x,y in zip(xs, ys):
			self.preds.append(Predator(x, y))
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
		prey_action = self.prey.take_action([self.prey.x, self.prey.y, min_pred_x-self.prey.x, min_pred_y-self.prey.y, self.basemap[self.prey.x, self.prey.y]])
		self.prey.record_action([self.prey.x, self.prey.y, min_pred_x-self.prey.x, min_pred_y-self.prey.y, self.basemap[self.prey.x, self.prey.y]], prey_action)
		communications = []
		for pred in self.preds:
			scream = pred.scream([pred.x, pred.y, self.prey.x-pred.x, self.prey.y-pred.y, self.basemap[pred.x, pred.y]])
			communications.append(scream)
		communications = np.array(communications)
		communications = np.mean(communications, axis=0)
		for pred in self.preds:
			pred_action = pred.take_action([pred.x, pred.y, self.prey.x-pred.x, self.prey.y-pred.y, self.basemap[pred.x, pred.y]]+[c for c in communications])
			pred.record_action([pred.x, pred.y, self.prey.x-pred.x, self.prey.y-pred.y, self.basemap[pred.x, pred.y]]+[c for c in communications], pred_action)

	def get_rewards(self):
		prey_x, prey_y = self.prey.x, self.prey.y
		d_arr = []
		for pred in self.preds:
			d = np.sqrt((prey_x - pred.x)**2 + (prey_y - pred.y)**2)/10
			d_arr.append(d)
		predator_rewards = [72.2-d for d in d_arr]
		prey_reward = min(d_arr)
		print(predator_rewards, prey_reward)

	def plot(self):
		colored_map = self.landscape.get_truemap()
		colored_map[self.prey.x-5:self.prey.x+5, self.prey.y-5:self.prey.y+5, :] = 0
		for pred in self.preds:
			colored_map[pred.x-7:pred.x+7, pred.y-7:pred.y+7, :] = 1
		#return colored_map
		cv2.imshow("", colored_map)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == '__main__':
	env = Environment()
	env.start()
	env.take_actions()
	env.get_rewards()
	# env.plot()