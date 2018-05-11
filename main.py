from copy import deepcopy
from csv import reader
import sys
import numpy as np
import random

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# random.seed(42) # for reproducibility

N_STATES = 4
N_EPISODES = 20

MAX_EPISODE_STEPS = 100

MIN_ALPHA = 0.02

alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
gamma = 1.0
eps = 0.2

q_table = dict()

ZOMBIE = "z"
CAR = "c"
ICE_CREAM = "i"
EMPTY = "*"

class State:
	
	def __init__(self, grid, car_pos):
		self.grid = grid
		self.car_pos = car_pos
		
	def __eq__(self, other):
		return isinstance(other, State) and self.grid == other.grid and self.car_pos == other.car_pos
	
	def __hash__(self):
		return hash(str(self.grid) + str(self.car_pos))
	
	def __str__(self):
		return "State(grid={self.grid}, car_pos={self.car_pos})"



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

def act(state, action):
	
	def new_car_pos(state, action):
		p = deepcopy(state.car_pos)
		if action == UP:
			p[0] = max(0, p[0] - 1)
		elif action == DOWN:
			p[0] = min(len(state.grid) - 1, p[0] + 1)
		elif action == LEFT:
			p[1] = max(0, p[1] - 1)
		elif action == RIGHT:
			p[1] = min(len(state.grid[0]) - 1, p[1] + 1)
		else:
			raise ValueError("Unknown action {action}")
		return p
			
	p = new_car_pos(state, action)
	grid_item = state.grid[p[0]][p[1]]
	
	new_grid = deepcopy(state.grid)
	
	if grid_item == ZOMBIE:
		reward = -100
		is_done = True
		new_grid[p[0]][p[1]] += CAR
	elif grid_item == ICE_CREAM:
		reward = 1000
		is_done = True
		new_grid[p[0]][p[1]] += CAR
	elif grid_item == EMPTY:
		reward = -1
		is_done = False
		old = state.car_pos
		new_grid[old[0]][old[1]] = EMPTY
		new_grid[p[0]][p[1]] = CAR
	elif grid_item == CAR:
		reward = -1
		is_done = False
	else:
		raise ValueError("Unknown grid item {grid_item}")
	
	return State(grid=new_grid, car_pos=p), reward, is_done

def q(state, action=None):
	
	if state not in q_table:
		q_table[state] = np.zeros(len(ACTIONS))
		
	if action is None:
		return q_table[state]
	
	return q_table[state][action]

def choose_action(state):
	if random.uniform(0, 1) < eps:
		return random.choice(ACTIONS) 
	else:
		return np.argmax(q(state))


#
# This is the main function of the program
#
if __name__ == "__main__":
	# Get the filename of the dataset
	filename = sys.argv[1]
	# Load the file into the dataset variable
	racetrack = load_csv(filename)

	grid = build_grid(racetrack)
	for row in grid:
		print ''.join(row)


	start_state = State(grid=grid, car_pos=[1, 1])

	for e in range(N_EPISODES):
	
		state = start_state
		total_reward = 0
		alpha = alphas[e]
		
		for _ in range(MAX_EPISODE_STEPS):
			action = choose_action(state)
			next_state, reward, done = act(state, action)
			total_reward += reward
			
			q(state)[action] = q(state, action) + alpha * (reward + gamma *  np.max(q(next_state)) - q(state, action))
			state = next_state
			if done:
				break
		print "Episode {" + str(e) + "}: total reward -> {" + str(total_reward) + "}"