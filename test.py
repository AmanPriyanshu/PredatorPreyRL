from env import Environment
import torch
import numpy as np
import cv2
from tqdm import trange

if __name__ == '__main__':
	np.random.seed(5)
	env = Environment()
	env.start()
	env.prey.model.load_state_dict(torch.load("prey_model.pt"))
	env.predator_model.load_state_dict(torch.load("pred_model.pt"))
	for i in trange(50):
		env.prey.epsilon = 1.0
		for pred in env.preds:
			pred.epsilon = 1.0
		env.take_actions()
		colored_map = env.plot(False)
		colored_map = colored_map*255
		cv2.imwrite("./imgs/img_"+"0"*(2-len(str(i)))+str(i)+".png", colored_map)