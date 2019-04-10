# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance, Queue
from game import Directions
import random, util
import operator
from game import Agent

class ReflexAgent(Agent):
	"""
	  A reflex agent chooses an action at each choice point by examining
	  its alternatives via a state evaluation function.

	  The code below is provided as a guide.  You are welcome to change
	  it in any way you see fit, so long as you don't touch our method
	  headers.
	"""

	def __init__(self):
		self.action2idx = { 'North' : (0, 1),
							'East'  : (1, 0),
							'South' : (0, -1),
							'West'  : (-1, 0),
							'Stop'  : (0, 0)}
		self.last_state = ()


	def _getDist2Ghost(self, newGhostStates, cur_pos):
		newGhostPos = []
		curGhostPos = []
		for ghostState in newGhostStates:
			ghost_curr_state = ghostState.getPosition()
			curGhostPos.append((int(ghost_curr_state[0]), int(ghost_curr_state[1])))
			ghost_next_move = self.action2idx[ghostState.getDirection()]
			ghost_next_state = (int(ghost_curr_state[0] + ghost_next_move[0]), int(ghost_curr_state[1] + ghost_next_move[1]))
			newGhostPos.append(ghost_next_state)
		d2g = [manhattanDistance(cur_pos, g_pos) for g_pos in newGhostPos]
		return min(d2g), curGhostPos


	def _getDist2Food(self, food_grid, cur_pos):
		all_food_pos = food_grid.asList()
		if len(all_food_pos) > 0:
			all_food_dist = [manhattanDistance(cur_pos, food_pos) for food_pos in all_food_pos]
			return min(all_food_dist)
		else: return 0


	def getAction(self, gameState):
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {North, South, West, East, Stop}
		"""

		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"
		self.last_state = gameState.getPacmanPosition()
		return legalMoves[chosenIndex]


	def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		"""
		Useful information you can extract from a GameState (pacman.py):
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
		"""
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()

		"*** YOUR CODE HERE ***"
		if action == 'Stop': return -100
		curPos = currentGameState.getPacmanPosition()
		curFood = currentGameState.getFood()
		
		new_d2g, curGhostPos = self._getDist2Ghost(newGhostStates, newPos)
		if newPos in curGhostPos: return -100
		
		new_d2f = self._getDist2Food(curFood, newPos)
		if newPos == self.last_state: 
			new_d2f += 3
		if new_d2g > 1:
			return  -new_d2f
		else:
			return -new_d2g if new_d2g > 0 else -100

	
def scoreEvaluationFunction(currentGameState):
	"""
	  This default evaluation function just returns the score of the state.
	  The score is the same one displayed in the Pacman GUI.

	  This evaluation function is meant for use with adversarial search agents
	  (not reflex agents).
	"""
	return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
	"""
	  This class provides some common elements to all of your
	  multi-agent searchers.  Any methods defined here will be available
	  to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	  You *do not* need to make any changes here, but you can if you want to
	  add functionality to all your adversarial search agents.  Please do not
	  remove anything, however.

	  Note: this is an abstract class: one that should not be instantiated.  It's
	  only partially specified, and designed to be extended.  Agent (game.py)
	  is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent (question 2)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action from the current gameState using self.depth
		  and self.evaluationFunction.

		  Here are some method calls that might be useful when implementing minimax.

		  gameState.getLegalActions(agentIndex):
			Returns a list of legal actions for an agent
			agentIndex=0 means Pacman, ghosts are >= 1

		  gameState.generateSuccessor(agentIndex, action):
			Returns the successor game state after an agent takes an action

		  gameState.getNumAgents():
			Returns the total number of agents in the game
		"""
		"*** YOUR CODE HERE ***"
		action = self.minimax(gameState, 0, self.index)
		return action


	def update_best(self, val, best_val, mode):
		ops = operator.ge if mode == 'max' else operator.le # >= or <=
		if ops(val, best_val): return val, True
		return best_val, False


	def minimax(self, gameState, cur_depth, Player):

		if cur_depth == self.depth:
			return self.evaluationFunction(gameState)

		if Player > 0:
			best_val = float('inf')
			actions = gameState.getLegalActions(Player)
			
			for action in actions:
				succ_state = gameState.generateSuccessor(Player, action)
				if gameState.getNumAgents() - 1 > Player: 
					val = self.minimax(succ_state, cur_depth, Player+1)
				else: 
					val = self.minimax(succ_state, cur_depth+1, self.index)
				best_val, update = self.update_best(val, best_val, mode='min') # take the least valuable action

			if len(actions) == 0: best_val = self.evaluationFunction(gameState)
			return best_val

		else:
			best_val = -float('inf')
			best_act = None
			actions = gameState.getLegalActions(Player)
			
			for action in actions:
				succ_state = gameState.generateSuccessor(self.index, action)
				val = self.minimax(succ_state, cur_depth, Player+1)
				best_val, update = self.update_best(val, best_val, mode='max') # take the most valuable action
				if update: best_act = action

			if len(actions) == 0: best_val = self.evaluationFunction(gameState)
			return best_val if cur_depth > 0 else best_act


class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"
		action = self.alpha_bete(gameState, 0, A=-float('inf'), B=float('inf'), Player=self.index)
		return action


	def alpha_bete(self, gameState, cur_depth, A, B, Player):

		if cur_depth == self.depth:
			return self.evaluationFunction(gameState)

		if Player > 0:
			actions = gameState.getLegalActions(Player)
			v = float('inf')

			for action in actions:
				succ_state = gameState.generateSuccessor(Player, action)
				if gameState.getNumAgents() - 1 > Player: 
					v = min(v, self.alpha_bete(succ_state, cur_depth, A, B, Player+1))
				else: 
					v = min(v, self.alpha_bete(succ_state, cur_depth+1, A, B, self.index))
				if A > v: return v
				B = min(v, B) # take the least valuable action

			if len(actions) == 0: v = self.evaluationFunction(gameState)
			return v

		else:
			actions = gameState.getLegalActions(Player)
			v = -float('inf')
			act = {}

			for action in actions:
				succ_state = gameState.generateSuccessor(self.index, action)
				v = max(v, self.alpha_bete(succ_state, cur_depth, A, B, Player+1))
				if v > B: return v
				A = max(v, A) # take the most valuable action
				if str(A) not in act: act[str(A)] = action

			if len(actions) == 0: v = self.evaluationFunction(gameState)
			if cur_depth > 0: return v
			else:
				sorted_A = sorted(act.keys(), key=lambda x: float(x), reverse=True)
				return act[sorted_A[0]]


class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""

	def getAction(self, gameState):
		"""
		  Returns the expectimax action using self.depth and self.evaluationFunction

		  All ghosts should be modeled as choosing uniformly at random from their
		  legal moves.
		"""
		"*** YOUR CODE HERE ***"
		action = self.expectiminimax(gameState, 0, self.index)
		return action


	def expectiminimax(self, gameState, cur_depth, Player):

		if cur_depth == self.depth:
			return self.evaluationFunction(gameState)

		if Player > 0:
			exp_val = 0
			actions = gameState.getLegalActions(Player)
			
			for action in actions:
				succ_state = gameState.generateSuccessor(Player, action)
				if gameState.getNumAgents() - 1 > Player: 
					val = self.expectiminimax(succ_state, cur_depth, Player+1)
				else: 
					val = self.expectiminimax(succ_state, cur_depth+1, self.index)
				exp_val += (1/float(len(actions)) * val)

			if len(actions) == 0: exp_val = self.evaluationFunction(gameState)
			return exp_val

		else:
			best_val = -float('inf')
			best_act = None
			actions = gameState.getLegalActions(Player)
			
			for action in actions:
				succ_state = gameState.generateSuccessor(self.index, action)
				val = self.expectiminimax(succ_state, cur_depth, Player+1)
				if val > best_val: best_act = action
				best_val = max(val, best_val) # take the most valuable action

			if len(actions) == 0: best_val = self.evaluationFunction(gameState)
			return best_val if cur_depth > 0 else best_act


def betterEvaluationFunction(currentGameState):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function (question 5).

	  DESCRIPTION: <write something here so we know what you did>
	"""
	"*** YOUR CODE HERE ***"
	util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

