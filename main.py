from copy import deepcopy
from csv import reader
import sys
import numpy as np
import random


# Arbitraty start
start_pos = [1,6]
# The actions you can take
ACTIONS = [-1, 0, 1]
# Number of episodes to do
N_EPISODES = 20
# All of the steps to take in an episode
MAX_EPISODE_STEPS = 100
# The alpha
MIN_ALPHA = 0.02
# All of the alphas to take
alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
# Gamma
gamma = 1.0
# To determine explore
eps = 0.2
# Teh table for the Q values
q_table = dict()
# What the race track is made of
WALL = "#"
CAR = "S"
FINISH = "F"
EMPTY = "."
# A Simple class to show state of the car
class State:
	# Set the state of the car
	def __init__(self, grid, car_pos, car_acc):
		# The grid you have in this state
		self.grid = grid
		# Where you are on the grid
		self.car_pos = car_pos
		# The acceleration of the car
		self.car_acc = car_acc
	# Checking for equality
	def __eq__(self, other):
		return isinstance(other, State) and self.grid == other.grid and self.car_pos == other.car_pos and self.car_acc == other.car_acc
	# The hash if you ever need it
	def __hash__(self):
		return hash(str(self.grid) + str(self.car_pos) + str(self.car_acc))
	# How to print yourself out
	def __str__(self):
		return "State(grid={self.grid}, car_pos={self.car_pos}, car_acc{self.car_acc})"



#
# This function will load in a CSV file
#
def load_csv(filename):
	# Create and empty list
	dataset = list()
	# Open the file to read
	with open(filename, 'r') as file:
		# Create a reader of the file
		csv_reader = reader(file)
		# For each row in the the file read in
		for row in csv_reader:
			# If there is an empty row
			if not row:
				# Skip
				continue
			# Add row the the dataset list
			dataset.append(row)
	# Return the dataset that you created
	return dataset
 

#
# This function will build the grid to be used
#
def build_grid(racetrack):
	# Start with an empty list
	grid = []
	# For each row in the racetrack passed in
	for row in racetrack:
		# Create and empty grid row
		grid_row = []
		# For each of the imporant character
		for i in range(2, len(str(row))-2):
			# Append them to the row
			grid_row.append(str(row)[i])
		# Append the finished row to the grid
		grid.append(grid_row)
	# Remove the first line
	del grid[0]
	# Return the grid
	return grid


#
# This function will determine what happens when an action is taken
#
def act(state, action):
	# Get a copy of the position you are in	
	p = deepcopy(state.car_pos)
	# Get the position of where you will be after the action
	y = action[0] + p[0]
	x = action[1] + p[1]
	# Set the position of wher eyou will be
	p = [y,x]
	# Get the item of where you will be on the gird
	grid_item = state.grid[y, x]
	# Get a copy of the grid
	new_grid = deepcopy(state.grid)
	# If there is a wall where you will be
	if grid_item == WALL:
		# Decrement the reward
		reward = -1
		# Put the car back at the starting position
		p = start_pos
		# Set velocity to 0
		acc = [0,0]
		# Put the car there for visuals
		new_grid[start_pos[0]][start_pos[1]] += CAR
		# You have not completed the race
		is_done = False
	# If you found the finish line
	elif grid_item == FINISH:
		# Decrement the reward
		reward = -1
		# You are done with the race
		is_done = True
		# Put the car there for visuals
		new_grid[p[0]][p[1]] += CAR
	# If you found an empty spot
	elif grid_item == EMPTY:
		# Decrement the reward
		reward = -1
		# You are not finished
		is_done = False
		# Get the position of the car when you started
		old = state.car_pos
		# Set the only posistion to be the empty
		new_grid[old[0]][old[1]] = EMPTY
		# Set the new position to be a car
		new_grid[y][x] = CAR
	# If you did not move
	elif grid_item == CAR:
		# Decrement the count
		reward = -1
		# You are not done with the race
		is_done = False
	else:
		raise ValueError("Unknown grid item {grid_item}")
	# Return the new state adn reward and if you are done or not
	return State(grid=new_grid, car_pos=p, car_acc=acc), reward, is_done


#
# This function will preform actions on the Q table
#
def q(state, action=None):
	# If the state you are in is not in the q table
	if state not in q_table:
		# Add it
		q_table[state] = state.car_acc
	# Set a return value
	ret = q_table[state]
	# If you gave no actions
	if action is None:
		# just set the return value to be what is in the table
		ret = q_table[state]
	# Return what is in the table for the state
	return q_table[state]

#
# This function will choose an action based on the state
#
def choose_action(state):
	# If you should explore randomly
	if random.uniform(0, 1) < eps:
		# Make a copy of where you are
		p = deepcopy(state.car_acc)
		# Pick a random action to take
		p[0] += random.choice(ACTIONS)
		# Make sure you do not go over the limit
		if p[0] > 5:
			p[0] = 5
		if p[0] < -5:
			p[0] = -5
		# Pick a random action to take
		p[1] += random.choice(ACTIONS)
		# Make sure you do not go over the limit
		if p[1] > 5:
			p[1] = 5
		if p[1] < -5:
			p[1] = -5
		# Return your action
		return p
	# Otherwise
	else:
		# You should return the best move you have seen so far
		return np.argmax(q(state))


#
# This is the main function of the program
#
if __name__ == "__main__":
	# Get the filename of the dataset
	filename = sys.argv[1]
	# Load the file into the dataset variable
	racetrack = load_csv(filename)
	# Build the grid
	grid = build_grid(racetrack)
	# For each row in the grid
	for row in grid:
		# Print out the row
		print ''.join(row)
	# Create a starting state
	start_state = State(grid=grid, car_pos=start_pos, car_acc=[0,0])
	# For each episode
	for e in range(N_EPISODES):
		# Set the state
		state = start_state
		# Set the reward for this episode
		total_reward = 0
		# Set the alpha for the episode
		alpha = alphas[e]
		# For all of the steps in the episodes
		for _ in range(MAX_EPISODE_STEPS):
			# Choose an action			
			action = choose_action(state)
			# Get the state, reward and if you are finished for this actions
			next_state, reward, done = act(state, action)
			# Add the reward to the running total
			total_reward += reward
			# Update the Q value based on the algorithm
			q(state)[action] = q(state, action) + alpha * (reward + gamma *  np.max(q(next_state)) - q(state, action))
			# set the state to the one you just reached
			state = next_state
			# If you reached the end
			if done:
				# Break
				break
		# print "Episode {" + str(e) + "}: total reward -> {" + str(total_reward) + "}"