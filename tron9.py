#They fight now. Top 10 winners are bread
#Can randomly add layers; doesn't 
#Winners survive. That's it. 



"""
Sample Python/Pygame Programs
Simpson College Computer Science
http://prograradcadegames.com/
http://simpson.edu/computer-science/
"""
import pygame
import numpy as np
from time import sleep
import random

pygame.init()
pygame.font.init() 

#Some colors for easy reference
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
L_RED = (255, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
L_BLUE = (200, 200, 255)
GREEN = (0, 255, 0)
GRAY = (150, 180, 180)
FILL = (200, 255, 255)
TEXT = BLACK

#some constants
WIDTH = 1600
HEIGHT = 700  

RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3

GRIDSIZE = (51, 51)
LAYER_STRUCTURE = [12, 4]

INDIVIDUALS = 100
NUMBER_OF_WINNERS = 10
NEW_LAYER_CHANCE = 0.0


#Refers to the boxes that comprise a mine sweeper board
class Box:
	def __init__(self, x1, x2, y1, y2, color=GRAY, index=0, free = True):
		#x1, y1, x2, y2 indicate the location of the box
		self.x1 = x1
		self.y1=y1
		self.x2=x2
		self.y2=y2
		#Tracks the color of the box
		self.color = color
		#index (which number box it is)
		self.index = index
		#Is the box free to move to, or is there a wall there?
		self.free = free
	
	#Draws the box onto the screen	
	def draw(self, color = "DEFAULT"):
		if color == "DEFAULT":
			pygame.draw.line(screen, self.color, [self.x1, self.y1], [self.x2, self.y1], boxSide)
		else:
			pygame.draw.line(screen, color, [self.x1, self.y1], [self.x2, self.y1], boxSide)

#Brain class
class Brain:
	#Initialize a neural network of a certain layer_structure
	def __init__(self, layer_structure, steps = 0, coefs = 0, intercepts = 0):
		self.steps = 0
		'''
		self.command = RIGHT
		'''
		self.layer_structure = layer_structure.copy()
		'''
		self.scorener1 = False
		'''
		if coefs == 0:
			self.coefs = self.generateCoefs(layer_structure)
		else:
			self.coefs = self.copyCoefs(coefs)
			
		if intercepts == 0:
			self.intercepts = self.generateIntercepts(layer_structure)
		else:
			self.intercepts = self.copyIntercepts(intercepts)
	
	#Returns a copy of a list of coefs. Creates a copy of each sub-array so the list is completely independent			
	def copyCoefs(self, coefs):
		newCoefs = []
		for array in coefs:
			newCoefs.append(np.copy(array))
		return newCoefs
	
	#Returns a copy of a list of intercepts. Creates a copy of each sub-array so the list is completely independent	
	def copyIntercepts(self, intercepts):
		newIntercepts = []
		for array in intercepts:
			newIntercepts.append(np.copy(array))
		return newIntercepts	
	
	#Creates a set of random coefficients between -1 and 1 for a given layer_structure
	def generateCoefs(self, layer_structure):
		coefs = []
		for i in range(len(layer_structure)-1):
			coefs.append(np.random.rand(layer_structure[i], layer_structure[i+1])*2-1)
		return coefs
	
	#Creates a set of random intercepts between -1 and 1 for a given layer_structure
	def generateIntercepts(self, layer_structure):
		intercepts = []
		for i in range(len(layer_structure)-1):
			intercepts.append(np.random.rand(layer_structure[i+1])*2-1)
		return intercepts
	
	#Returns a set of coefficients which are a slight mutation of its own (normal distribution)
	def mutateCoefs(self):
		newCoefs = self.copyCoefs(self.coefs)
		
		for i in range(len(newCoefs)):
			for row in range(len(newCoefs[i])):
				for col in range(len(newCoefs[i][row])):
					newCoefs[i][row][col] = np.random.normal(newCoefs[i][row][col], 1)
		return newCoefs
	
	#Returns a set of intercepts which are a slight mutation of its own (normal distribution)
	def mutateIntercepts(self):
		newIntercepts = self.copyIntercepts(self.intercepts)

		for i in range(len(newIntercepts)):
			for row in range(len(newIntercepts[i])):
				newIntercepts[i][row] = np.random.normal(newIntercepts[i][row], 1)
		return newIntercepts
	
	#Returns a new brain, which is a itself but with mutated coefs, intercepts, and potentially layer_structure	
	def mutate(self):
		newBrain = Brain(layer_structure = self.layer_structure.copy(), coefs = self.mutateCoefs(), intercepts = self.mutateIntercepts())
		
		if random.random() < NEW_LAYER_CHANCE:
			newBrain.addLayer()
		
		return newBrain
	
	#Duplicates a player in the neural network, and connects it to the duplicated layer with weights of 1
	def addLayer(self):
		index = random.randint(1,len(self.layer_structure))
		self.layer_structure.insert(index, self.layer_structure[index-1])
		self.coefs.insert(index-1, np.identity(self.layer_structure[index-1]))
		self.intercepts.insert(index-1, np.zeros((self.layer_structure[index-1])))

	#Predicts the output of a brain for a given input
	def calculateOutput(self, input, g="identity"):
		#(Stuff is transposed since we need columns for matrix multiplication)
		#The values of the neurons for each layer will be stored in "layers", so here the input layer is added to start
		layers = [np.transpose(input)]
		#The current layer will be affected by the previous layer, so here we define the initial previousLayer to be the input 
		previousLayer = np.transpose(input)
	
		reduced_layer_structure = self.layer_structure[1:]
		
		#Loops through all the layers, excluding the first 
		for k in range(len(reduced_layer_structure)):
			#creates an empty array of the correct size
			currentLayer = np.empty((reduced_layer_structure[k],1))
			#The resulting layer is a matrix multiplication of the previousLayer and the coefficients, plus the self.intercepts
			result = np.matmul(np.transpose(self.coefs[k]),previousLayer) + np.transpose(np.array([self.intercepts[k]]))
			#The value of each neuron is then put through a function g()
			for i in range(len(currentLayer)):
				if g == "identity":
					currentLayer[i] = result[i]
				elif g == "relu":
					currentLayer[i] = max(0, result[i])
				elif g == "tanh":
					currentLayer[i] = tanh(result[i])
				elif g == "logistic":
					try:
						currentLayer[i] = 1 / (1 + exp(-1*result[i]))
					except OverflowError:
						currentLayer[i] = 0
			#The current layer is then added to the layers list, and the previousLayer variable is updated
			layers.append(currentLayer)
			previousLayer = currentLayer.copy()
	
		#Returns the index of the highest value neuron in the output layer (aka layers[-1])
		#E.g. if the 7th neuron has the highest value, returns 7
		return(layers[-1].tolist().index(max(layers[-1].tolist())))	
		
#--------------------------------------Functions-------------------------------------------

#Translate the number of a box in array "boxes" to row and column data
def numtoRC(index):
	row = index / GRIDSIZE[1]
	col = index % GRIDSIZE[1]
	return (row,col)
	
#Translates the row and column data of a box into the number of the box in array "boxes"
def RCtoNum(RC_list):
	row = RC_list[0]
	col = RC_list[1]
	return GRIDSIZE[1]*row + col

#Creates and returns the list of boxes
def createBoxes():
	boxes = []
	index = 0
	for j in range(GRIDSIZE[0]):
		for i in range(GRIDSIZE[1]):
			boxes.append(Box(x1=int(Xoffset+i*boxSide), y1=int(Yoffset+j*boxSide-boxSide/2), x2=int(Xoffset+(i+1)*boxSide), y2=int(Yoffset+j*boxSide+boxSide/2), color=GRAY, index = index))
			index += 1
	return boxes			

def score_and_reset(boxes, localWinner, localLoser, hitBorder = False):
	global P1Position
	global P2Position
	global P1Direction
	global P2Direction
	global P1Wins
	global P2Wins
	global walls
	
	#Winners get a score of 1
	'''
	if localWinner.steps > int(np.ceil(GRIDSIZE[1]/2)):
		localWinner.score = 1
	else:
		localWinner.score = 0
	#Losers get a score of 0
	localLoser.score = 0
	'''
	
	if P1Direction == RIGHT and P2Direction == LEFT:
		localWinner.score = 0
		localLoser.score = 0
	else:
		#Winner incentivized to kill asap
		localWinner.score = GRIDSIZE[0]*GRIDSIZE[1] - localWinner.steps
		#Losers gets rewarded for staying alive
		localLoser.score = localLoser.steps
	
		if localWinner.team == 1:
			P1Wins += 1
		else:
			P2Wins += 1
		
	for i, box in enumerate(boxes):
		if box.color == L_BLUE or box.color == BLUE or box.color == L_RED or box.color == RED:
			box.color = GRAY	
			box.free=True
			walls[i] = 0
	
	#reset position and direction to starting position and direction
	P1Position = [int(GRIDSIZE[0]/2), 1]	
	P2Position = [int(GRIDSIZE[0]/2), int(GRIDSIZE[1]-2)]
	P1Direction = RIGHT
	P2Direction = LEFT
		

#Displays the nodes of a network, along with weighted lines showing the coefficient influences
def displayNetwork(screen, layer_structure, coefs):
	
	#Stores the larges coefficient, so we can scale the thicknesses accordingly. 
	max_coef = np.max(coefs[0])
	
	#Determines how much space this visual network will take up
	height = 700
	width = 300
	
	inputs = ["distDown", "distUp", "distLeft", "distRight", "distDownRight", "distUpRight", "distDownLeft", "distUpLeft", "self X", "self Y", "enemy X", "enemy Y"]
	
	outputs = ["right", "up", "left", "down"]
	
	
	layerCount = len(layer_structure)
	#This will store the positions of all the nodes (organized with sub-lists of each layer)
	circle_positions = []
	
	
	#Label inputs
	for i in range(layer_structure[0]):
		font= pygame.font.SysFont('Calibri', 30, False, False)
		text = font.render(inputs[i], True, TEXT)
		screen.blit(text,[0,(i+1)* int(height/(layer_structure[0]+2))])	
	
	#Label outputs
	for i in range(layer_structure[-1]):
		font= pygame.font.SysFont('Calibri', 30, False, False)
		text = font.render(str(outputs[i]), True, TEXT)
		screen.blit(text,[width+50,(i+1)* int(height/(layer_structure[-1]+2))])	
	
	#Calculates an appropriate spacing of the layers
	xspacing = int( width/(layerCount))
	
	#Determine the location of the neurons for each layer, stores that in a list, and stores those lists in circle_positions
	for i in range(layerCount):
		layer_circle_positions = []
		yspacing = max(6, int( height/(layer_structure[i]+2)))
		for j in range(layer_structure[i]):
			layer_circle_positions.append(((i+1)*xspacing, (j+1)*yspacing))
		circle_positions.append(layer_circle_positions)
	
	#Draws a line between every node in one layer and every node in the next layer
	for i in range(len(circle_positions)-1):
		for j, circle_pos in enumerate(circle_positions[i]):
			for k, circle_pos2 in enumerate(circle_positions[i+1]):
				thickness = int(coefs[i][j,k]/max_coef*8)
				
				if thickness > 0:
					pygame.draw.lines(screen, BLUE, False, [circle_pos, circle_pos2], thickness)
				else:
					pygame.draw.lines(screen, RED, False, [circle_pos, circle_pos2], -thickness)
					

	#Draws circles in the positions of the nodes (over the lines)
	for layer in circle_positions:
		for circle_pos in layer:
			pygame.draw.circle(screen, BLACK, circle_pos, 20, 0)
			pygame.draw.circle(screen, GREEN, circle_pos, 16, 0)

def nearSighted(distance):
	if np.abs(distance) >= 3:
		return 0
	else:
		return (4-np.abs(distance))
							
#--------------------------------------Program Setup-------------------------------------------

# Set the width and height of the screen [width,height]
size = [WIDTH, HEIGHT]

screen = pygame.display.set_mode(size)
 
 #Title
#pygame.display.set_caption("Win -> score=1. Lose -> score=0")
pygame.display.set_caption("winner.score = GRIDSIZE[0]*GRIDSIZE[1] - steps")

# Used to manage how fast the screen updates
clock = pygame.time.Clock()
 
# Hide the mouse cursor
pygame.mouse.set_visible(1)


# ---------------------------------- Game Setup -----------------------------------
#Gives each player an initial direction
P1Direction = RIGHT
P2Direction = LEFT

#Stores the number of wins for each player
P1Wins = 0
P2Wins = 0

#Gives each player an initial position
P1Position = [int(GRIDSIZE[0]/2), 1]
P2Position = [int(GRIDSIZE[0]/2), int(GRIDSIZE[1]-2)]

#Defines box dimensions and spacing dynamically, based on the size of the screen
boxSide = int(min(WIDTH/(GRIDSIZE[1]+2), HEIGHT/(GRIDSIZE[0]+2)))
Xoffset = int((WIDTH - (GRIDSIZE[1]*boxSide)) / 2)
Yoffset = int((HEIGHT - (GRIDSIZE[0]*boxSide)) / 2)

#Creates the boxes (grid)
boxes = createBoxes()
#Creates the wall array, which stores which spaces (aka boxes) have walls (1) and which don't (0)
walls = np.zeros((GRIDSIZE[0])*(GRIDSIZE[1]))

#Creates a border of walls around the screen 
for i in range(GRIDSIZE[0]):
	for j in range(GRIDSIZE[1]):
		if i == 0 or j==0 or i == GRIDSIZE[0]-1 or j == GRIDSIZE[1]-1:
			walls[RCtoNum([i,j])] = 1
			boxes[RCtoNum([i,j])].free = False
			boxes[RCtoNum([i,j])].color = BLACK

# ---------------------------------- Main Loop -----------------------------------

generation = 1
exit = False

#Loops through generations, in which 2 sets of INDIVIDUALS(say, 50) drivers compete against one another
#The best NUMBER_OF_WINNERS(say, 10) drivers are selected to reproduce (copy and mutate)
while exit == False:
	scores1 = []
	scores2 = []
	brains1 = []
	brains2 = []
	
	steps_high_score1 = -9e99
	steps_high_score2 = -9e99
	
	#If this is the first generation, then we create two sets of new drivers
	if generation == 1:
		for i in range(INDIVIDUALS):
			brains1.append(Brain(LAYER_STRUCTURE))
			brains2.append(Brain(LAYER_STRUCTURE))
	#Otherwise we mutate the winners from last round
	else:
		for j in range(NUMBER_OF_WINNERS):
			for i in range(int(INDIVIDUALS/NUMBER_OF_WINNERS)):
				brains1.append(winners1[j].mutate())
				brains2.append(winners2[j].mutate())
	
	for brain in brains1:
		brain.team = 1
	for brain in brains2:
		brain.team = 2
		
	#The drivers are shuffled so that the same drivers aren't always competing against each other
	random.shuffle(brains1)
	random.shuffle(brains2)
	print("Generation = " + str(generation))
	
	#Here we loop through all of the drivers from each team, one at a time from each (the i'th), and they compete
	for i in range(len(brains1)):
		
		#This is the battle between two individuals; the i'th from team1, and the i'th from team2
		done = False			
		while done == False:
			#The user can exit at any time
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					done = True
					exit = True
					break
			
			#The driver's brain needs to make a decision about which direction to go. This is the output. 
			#In order to calculate an output, we need some inputs:
			
			#x and y position of driver 1
			x1 = P1Position[0]
			y1 = P1Position[1]
			
			#x and y position of driver 2
			x2 = P2Position[0]
			y2 = P2Position[1]
			
			#Distance to nearest box to the right
			for j in range(1, GRIDSIZE[0]-P1Position[0]):
				if walls[RCtoNum([P1Position[0]+j, P1Position[1]])]:
					distDown1 = nearSighted(j)
					break
			#Distance to nearest box to the left
			for j in range(1, GRIDSIZE[0]):
				if walls[RCtoNum([P1Position[0]-j, P1Position[1]])]:
					distUp1 = nearSighted(j)
					break
			#Distance to nearest box down
			for j in range(1, GRIDSIZE[1]-P1Position[1]):
				if walls[RCtoNum([P1Position[0], P1Position[1]+j])]:
					distRight1 = nearSighted(j)
					break
			#Distance to nearest box up
			for j in range(1, GRIDSIZE[1]):
				if walls[RCtoNum([P1Position[0], P1Position[1]-j])]:
					distLeft1 = nearSighted(j)
					break
			#Distance to nearest box to the downRight
			for j in range(1, max(GRIDSIZE[0],GRIDSIZE[1])):
				if walls[RCtoNum([P1Position[0]+j, P1Position[1]+j])]:
					distDownRight1 = nearSighted(j)
					#boxes[RCtoNum([P1Position[0]+j, P1Position[1]+j])].color = GREEN
					break
			#Distance to nearest box to the downLeft
			for j in range(1, max(GRIDSIZE[0],GRIDSIZE[1])):
				if walls[RCtoNum([P1Position[0]+j, P1Position[1]-j])]:
					distDownLeft1 = nearSighted(j)
					#boxes[RCtoNum([P1Position[0]+j, P1Position[1]-j])].color = GREEN
					break
			#Distance to nearest box to the upRight
			for j in range(1, max(GRIDSIZE[0],GRIDSIZE[1])):
				if walls[RCtoNum([P1Position[0]-j, P1Position[1]+j])]:
					distUpRight1 = nearSighted(j)
					#boxes[RCtoNum([P1Position[0]-j, P1Position[1]+j])].color = GREEN
					break
			#Distance to nearest box to the upLeft
			for j in range(1, max(GRIDSIZE[0],GRIDSIZE[1])):
				if walls[RCtoNum([P1Position[0]-j, P1Position[1]-j])]:
					distUpLeft1 = nearSighted(j)
					#boxes[RCtoNum([P1Position[0]-j, P1Position[1]-j])].color = GREEN
					break		
			#Distance to nearest box to the right
			for j in range(1, GRIDSIZE[0]-P2Position[0]):
				if walls[RCtoNum([P2Position[0]+j, P2Position[1]])]:
					distDown2 = nearSighted(j)
					break
			#Distance to nearest box to the left
			for j in range(1, GRIDSIZE[0]):
				if walls[RCtoNum([P2Position[0]-j, P2Position[1]])]:
					distUp2 = nearSighted(j)
					break
			#Distance to nearest box down
			for j in range(1, GRIDSIZE[1]-P2Position[1]):
				if walls[RCtoNum([P2Position[0], P2Position[1]+j])]:
					distRight2 = nearSighted(j)
					break
			#Distance to nearest box up
			for j in range(1, GRIDSIZE[1]):
				if walls[RCtoNum([P2Position[0], P2Position[1]-j])]:
					distLeft2 = nearSighted(j)
					break
			#Distance to nearest box to the downRight
			for j in range(1, max(GRIDSIZE[0],GRIDSIZE[1])):
				if walls[RCtoNum([P2Position[0]+j, P2Position[1]+j])]:
					distDownRight2 = nearSighted(j)
					#boxes[RCtoNum([P2Position[0]+j, P2Position[1]+j])].color = GREEN
					break
			#Distance to nearest box to the downLeft
			for j in range(1, max(GRIDSIZE[0],GRIDSIZE[1])):
				if walls[RCtoNum([P2Position[0]+j, P2Position[1]-j])]:
					distDownLeft2 = nearSighted(j)
					#boxes[RCtoNum([P2Position[0]+j, P2Position[1]-j])].color = GREEN
					break
			#Distance to nearest box to the upRight
			for j in range(1, max(GRIDSIZE[0],GRIDSIZE[1])):
				if walls[RCtoNum([P2Position[0]-j, P2Position[1]+j])]:
					distUpRight2 = nearSighted(j)
					#boxes[RCtoNum([P2Position[0]-j, P2Position[1]+j])].color = GREEN
					break
			#Distance to nearest box to the upLeft
			for j in range(1, max(GRIDSIZE[0],GRIDSIZE[1])):
				if walls[RCtoNum([P2Position[0]-j, P2Position[1]-j])]:
					distUpLeft2 = nearSighted(j)
					#boxes[RCtoNum([P2Position[0]-j, P2Position[1]-j])].color = GREEN
					break
					
			'''
			print("\ndistLeft1 = " + str(distLeft1))
			print("distRight1 = " + str(distRight1))
			print("\ndistUp1 = " + str(distLeft1))
			print("distDown1 = " + str(distRight1))
			print("\ndistDownRight1 = " + str(distDownRight1))
			print("distDownLeft1 = " + str(distDownLeft1))
			print("\ndistUpLeft1 = " + str(distUpLeft1))
			print("distUpRight1 = " + str(distUpRight1))
			print("\ndistLeft2 = " + str(distLeft2))
			print("distRight2 = " + str(distRight2))
			print("\ndistUp2 = " + str(distLeft2))
			print("distDown2 = " + str(distRight2))
			print("\ndistDownRight2 = " + str(distDownRight2))
			print("distDownLeft2 = " + str(distDownLeft2))
			print("\ndistUpLeft2 = " + str(distUpLeft2))
			print("distUpRight2 = " + str(distUpRight2))
			'''

			#Here we insert all of those input variables into one list 
			input1 = np.array([[distDown1]+[distUp1]+[distLeft1]+[distRight1]+[distDownRight1]+[distUpRight1]+[distDownLeft1]+[distUpLeft1] + [x1] + [y1] + [x2] + [y2]])
			input2 = np.array([[distDown2]+[distUp2]+[distLeft2]+[distRight2]+[distDownRight2]+[distUpRight2]+[distDownLeft2]+[distUpLeft2] + [x2] + [y2] + [x1] + [y1]])
			#input1 = np.array([[distDown1]+[distUp1]+[distLeft1]+[distRight1]+[distDownRight1]+[distUpRight1]+[distDownLeft1]+[distUpLeft1]])
			#input2 = np.array([[distDown2]+[distUp2]+[distLeft2]+[distRight2]+[distDownRight2]+[distUpRight2]+[distDownLeft2]+[distUpLeft2]])			
			#input1 = np.array([[distDown1]+[distUp1]+[distLeft1]+[distRight1] + [x1] + [y1] + [x2] + [y2]])
			#input2 = np.array([[distDown2]+[distUp2]+[distLeft2]+[distRight2]+ [x2] + [y2] + [x1] + [y1]])
			
			#Now the drivers' brains can calculate which direction to go
			P1Direction = brains1[i].calculateOutput(input1)
			P2Direction = brains2[i].calculateOutput(input2)
			
			#The space/box in that direction of the player is checked to see if there is a wall
			if P1Direction == LEFT: 
				#If there is a wall there, the score is modified and the battle ends. 
				if boxes[RCtoNum([P1Position[0], P1Position[1]-1])].free == False:
					if boxes[RCtoNum([P1Position[0], P1Position[1]-1])].color == BLACK: #Checks to see if the crash was with the border wall
						score_and_reset(boxes, localWinner=brains2[i], localLoser=brains1[i], hitBorder = True)
						done = True
					else:
						score_and_reset(boxes, localWinner=brains2[i], localLoser=brains1[i])
						done = True
				#otherwise the driver moves to that position
				else:
					P1Position[1] -= 1
					
			elif P1Direction == RIGHT: 
				if boxes[RCtoNum([P1Position[0], P1Position[1]+1])].free == False:
					if boxes[RCtoNum([P1Position[0], P1Position[1]+1])].color == BLACK:
						score_and_reset(boxes, localWinner=brains2[i], localLoser=brains1[i], hitBorder = True)
						done = True
					else:
						score_and_reset(boxes, localWinner=brains2[i], localLoser=brains1[i])
						done = True
				else:
					P1Position[1] += 1
					
			elif P1Direction == UP:
				if boxes[RCtoNum([P1Position[0]-1, P1Position[1]])].free == False:
					if boxes[RCtoNum([P1Position[0]-1, P1Position[1]])].color == BLACK:
						score_and_reset(boxes, localWinner=brains2[i], localLoser=brains1[i], hitBorder = True)
						done = True
					else:
						score_and_reset(boxes, localWinner=brains2[i], localLoser=brains1[i])
						done = True
				else:
					P1Position[0] -= 1
					
			elif P1Direction == DOWN:
				if boxes[RCtoNum([P1Position[0]+1, P1Position[1]])].free == False:
					if boxes[RCtoNum([P1Position[0]+1, P1Position[1]])].color == BLACK:
						score_and_reset(boxes, localWinner=brains2[i], localLoser=brains1[i], hitBorder = True)
						done = True
					else:
						score_and_reset(boxes, localWinner=brains2[i], localLoser=brains1[i])
						done = True
				else:
					P1Position[0] += 1
			
			if P2Direction == LEFT: 
				if boxes[RCtoNum([P2Position[0], P2Position[1]-1])].free == False:
					if boxes[RCtoNum([P2Position[0], P2Position[1]-1])].color == BLACK:
						score_and_reset(boxes, localWinner=brains1[i], localLoser=brains2[i], hitBorder = True)
						done = True
					else:
						score_and_reset(boxes, localWinner=brains1[i], localLoser=brains2[i])
						done = True
				else:
					P2Position[1] -= 1
					
			elif P2Direction == RIGHT: 
				if boxes[RCtoNum([P2Position[0], P2Position[1]+1])].free == False:
					if boxes[RCtoNum([P2Position[0], P2Position[1]+1])].color == BLACK:
						score_and_reset(boxes, localWinner=brains1[i], localLoser=brains2[i], hitBorder = True)
						done = True
					else:
						score_and_reset(boxes, localWinner=brains1[i], localLoser=brains2[i])
						done = True
				else:
					P2Position[1] += 1
					
			elif P2Direction == UP:
				if boxes[RCtoNum([P2Position[0]-1, P2Position[1]])].free == False:
					if boxes[RCtoNum([P2Position[0]-1, P2Position[1]])].color == BLACK:
						score_and_reset(boxes, localWinner=brains1[i], localLoser=brains2[i], hitBorder = True)
						done = True
					else:
						score_and_reset(boxes, localWinner=brains1[i], localLoser=brains2[i])
						done = True
				else:
					P2Position[0] -= 1
					
			elif P2Direction == DOWN:
				if boxes[RCtoNum([P2Position[0]+1, P2Position[1]])].free == False:
					if boxes[RCtoNum([P2Position[0]+1, P2Position[1]])].color == BLACK:
						score_and_reset(boxes, localWinner=brains1[i], localLoser=brains2[i], hitBorder = True)
						done = True
					else:
						score_and_reset(boxes, localWinner=brains1[i], localLoser=brains2[i])
						done = True
				else:
					P2Position[0] += 1
			
			#Each brain is awarded one point for making it through the turn (even if it is their last)		
			brains1[i].steps += 1
			if brains1[i].steps > steps_high_score1:
				steps_high_score1 = brains1[i].steps
			
			brains2[i].steps += 1
			if brains2[i].steps > steps_high_score2:
				steps_high_score2 = brains2[i].steps
				
			#The space currently occupied by the player is colored and marked as a wall
			boxes[RCtoNum(P1Position)].color = L_RED
			boxes[RCtoNum(P2Position)].color = L_BLUE
			
			boxes[RCtoNum(P1Position)].free = False
			boxes[RCtoNum(P2Position)].free = False
			
			walls[RCtoNum(P1Position)] = 1
			walls[RCtoNum(P2Position)] = 1
			
			#DRAW
			screen.fill(FILL)
			
			for box in boxes:
				box.draw()
			boxes[RCtoNum(P1Position)].draw(RED)
			boxes[RCtoNum(P2Position)].draw(BLUE)
			
			displayNetwork(screen, brains1[i].layer_structure, brains1[i].coefs)
			
			#score board
			font= pygame.font.SysFont('Calibri', 50, False, False)
			
			text = font.render("Generation = " + str(generation), True, TEXT)
			screen.blit(text,[int(size[0]/2)-100,40])	
			text2 = font.render("Battle = " + str(i+1) + "/" + str(INDIVIDUALS), True, TEXT)
			screen.blit(text2,[int(size[0]/2)-100,70])		
			
			text3 = font.render("Steps = " + str(brains1[i].steps), True, TEXT)
			screen.blit(text3,[500,size[1]-30])		

			
			text7 = font.render("P1 Wins = " + str(P1Wins), True, TEXT)
			screen.blit(text7,[50,size[1]-60])		
			text8 = font.render("P2 Wins = " + str(P2Wins), True, TEXT)
			screen.blit(text8,[50,size[1]-30])	
				
			pygame.display.flip()
		
			clock.tick(120)
	
			if done:
				break
	
		if exit == True:
			break
		
		
		#The scores are added to a list of scores, so that the highest scores can be decided later
		scores1.append(brains1[i].score)
		scores2.append(brains2[i].score)	
				
	#Select [NUMBER_OF_WINNERS](say, 10) winners from team1
	topIndexes1 = sorted(range(len(scores1)), key=lambda i: scores1[i])[-(NUMBER_OF_WINNERS):]
	winners1 = []
	for index in topIndexes1:
		winners1.append(brains1[index])

	
	#find top [NUMBER_OF_WINNERS](say, 10) Blue winners from team2 at the end of each generation + some rando's
	topIndexes2 = sorted(range(len(scores2)), key=lambda i: scores2[i])[-(NUMBER_OF_WINNERS):]
	winners2 = []
	for index in topIndexes2:
		winners2.append(brains2[index])
	
	#Print some info to screen
	print("Generation=" + str(generation))

	generation += 1
	
# Close the window and quit.
pygame.quit()