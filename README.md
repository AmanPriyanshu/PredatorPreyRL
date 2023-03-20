# PredatorPreyRL
Example of Multi-Agent Communication in RL systems

![demo](/demo.gif)

## Reinforcement Learning (RL)

Reinforcement Learning (RL) is a type of Machine Learning in which an agent learns to interact with its environment by taking actions and receiving rewards. The agent learns to maximize its rewards by learning a policy that maps states to actions. While there are a multitude of ways to go about executing RL, model-free RL has become popular primarily due to its use of a Deep Neural Network to map states and actions directly. This allows learning over complex environments and employs the predictive capability of DNNs, thus enabling better performance without a complexity tradeoff.

## Multi-Agent RL (MARL)

Multi-Agent RL is an extension of RL in which multiple agents interact with each other and their environment simultaneously. This enables more complex and real-world like simulations, as well as the ability for agents to learn from each other's experiences. This type of RL can be used to solve a range of tasks, from robotics to game playing. 

## Predator-Prey Problem

The Predator Prey Problem is a classic example of MARL, in which two or more agents (the predator(s) and the prey) interact with each other and their environment in order to maximize their respective rewards. It is inspired by real-world attributes and natural events of predation. Typically the predator's reward is to capture the prey, while the prey's reward is to avoid such capture. At the same time this particular implementation looks at employing communication between the predators to maximize their rewards.
