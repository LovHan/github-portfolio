# myTeam.py
# ---------------
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


# myTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from ast import Raise
from typing import List, Tuple

from numpy import true_divide
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent
from util import nearestPoint
import sys,os

# the folder of current file.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

from lib_piglet.utils.pddl_solver import pddl_solver
from lib_piglet.domains.pddl import pddl_state
from lib_piglet.utils.pddl_parser import Action

CLOSE_DISTANCE = 4
MEDIUM_DISTANCE = 15
LONG_DISTANCE = 25


#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
                             first = 'MixedAgent', second = 'MixedAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########                                       

class MixedAgent(CaptureAgent):
    """
    This is an agent that use pddl to guide the high level actions of Pacman
    """
    # Default weights for q learning, if no QLWeights.txt find, we use the following weights.
    # You should add your weights for new low level planner here as well.
    # weights are defined as class attribute here, so taht agents share same weights.
    QLWeights = {
            "offensiveWeights":{'closest-food': -1, 
                                        'bias': 1, 
                                        '#-of-ghosts-1-step-away': -100, 
                                        'successorScore': 100, 
                                        'chance-return-food': 10,
                                        },
            "defensiveWeights": {'numInvaders': -1000, 'onDefense': 100,'teamDistance':2 ,'invaderDistance': -10, 'stop': -100, 'reverse': -2},
            "escapeWeights": {'onDefense': 1000, 'enemyDistance': 30, 'stop': -100, 'distanceToHome': -20}
        }
    QLWeightsFile = BASE_FOLDER+'/QLWeightsMyTeam.txt'

    # Also can use class variable to exchange information between agents.
    CURRENT_ACTION = {}


    def registerInitialState(self, gameState: GameState):
        self.pddl_solver = pddl_solver(BASE_FOLDER+'/myTeam.pddl')
        self.highLevelPlan: List[Tuple[Action,pddl_state]] = None # Plan is a list Action and pddl_state
        self.currentNegativeGoalStates = []
        self.currentPositiveGoalStates = []
        self.currentActionIndex = 0 # index of action in self.highLevelPlan should be execute next

        self.startPosition = gameState.getAgentPosition(self.index) # the start location of the agent
        CaptureAgent.registerInitialState(self, gameState)

        self.lowLevelPlan: List[Tuple[str,Tuple]] = []
        self.lowLevelActionIndex = 0

        # REMEMBER TRUN TRAINNING TO FALSE when submit to contest server.
        self.trainning = False # trainning mode to true will keep update weights and generate random movements by prob.
        self.epsilon = 0.1 #default exploration prob, change to take a random step
        self.alpha = 0.02 #default learning rate
        self.discountRate = 0.9 # default discount rate on successor state q value when update
        
        # Use a dictionary to save information about current agent.
        MixedAgent.CURRENT_ACTION[self.index]={}
        """
        Open weights file if it exists, otherwise start with empty weights.
        NEEDS TO BE CHANGED BEFORE SUBMISSION

        """
        if os.path.exists(MixedAgent.QLWeightsFile):
            with open(MixedAgent.QLWeightsFile, "r") as file:
                MixedAgent.QLWeights = eval(file.read())
            print("Load QLWeights:",MixedAgent.QLWeights )
        
    
    def final(self, gameState : GameState):
        """
        This function write weights into files after the game is over. 
        You may want to comment (disallow) this function when submit to contest server.
        """
        if self.trainning:
            print("Write QLWeights:", MixedAgent.QLWeights)
            file = open(MixedAgent.QLWeightsFile, 'w')
            file.write(str(MixedAgent.QLWeights))
            file.close()
    

    def chooseAction(self, gameState: GameState):
        """
        This is the action entry point for the agent.
        In the game, this function is called when its current agent's turn to move.

        We first pick a high-level action.
        Then generate low-level action ("North", "South", "East", "West", "Stop") to achieve the high-level action.
        """

        #-------------High Level Plan Section-------------------
        # Get high level action from a pddl plan.

        # Collect objects and init states from gameState
        objects, initState = self.get_pddl_state(gameState)
        positiveGoal, negtiveGoal = self.getGoals(objects,initState)

        # Check if we can stick to current plan 
        if not self.stateSatisfyCurrentPlan(initState, positiveGoal, negtiveGoal):
            # Cannot stick to current plan, prepare goals and replan
            print("Agnet:",self.index,"compute plan:")
            print("\tOBJ:"+str(objects),"\tINIT:"+str(initState), "\tPOSITIVE_GOAL:"+str(positiveGoal), "\tNEGTIVE_GOAL:"+str(negtiveGoal),sep="\n")
            self.highLevelPlan: List[Tuple[Action,pddl_state]] = self.getHighLevelPlan(objects, initState,positiveGoal, negtiveGoal) # Plan is a list Action and pddl_state
            self.currentActionIndex = 0
            self.lowLevelPlan = [] # reset low level plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            print("\tPLAN:",self.highLevelPlan)
        if len(self.highLevelPlan)==0:
            raise Exception("Solver retuned empty plan, you need to think how you handle this situation or how you modify your model ")
        
        # Get next action from the plan
        highLevelAction = self.highLevelPlan[self.currentActionIndex][0].name
        MixedAgent.CURRENT_ACTION[self.index] = highLevelAction
        print("Agent:", self.index, highLevelAction)

        #-------------Low Level Plan Section-------------------
        # Get the low level plan using Q learning, and return a low level action at last.
        # A low level action is defined in Directions, whihc include {"North", "South", "East", "West", "Stop"}

        if not self.posSatisfyLowLevelPlan(gameState):
            self.lowLevelPlan = self.getLowLevelPlanHS(gameState, highLevelAction) #Generate low level plan with q learning
            # you can replace the getLowLevelPlanQL with getLowLevelPlanHS and implement heuristic search planner
            self.lowLevelActionIndex = 0
        lowLevelAction = self.lowLevelPlan[self.lowLevelActionIndex][0]
        self.lowLevelActionIndex+=1
        print("\tAgent:", self.index,lowLevelAction)
        return lowLevelAction

    #------------------------------- PDDL and High-Level Action Functions ------------------------------- 
    
    
    def getHighLevelPlan(self, objects, initState, positiveGoal, negtiveGoal) -> List[Tuple[Action,pddl_state]]:
        """
        This function prepare the pddl problem, solve it and return pddl plan
        """
        # Prepare pddl problem
        self.pddl_solver.parser_.reset_problem()
        self.pddl_solver.parser_.set_objects(objects)
        self.pddl_solver.parser_.set_state(initState)
        self.pddl_solver.parser_.set_negative_goals(negtiveGoal)
        self.pddl_solver.parser_.set_positive_goals(positiveGoal)
        
        # Solve the problem and return the plan
        return self.pddl_solver.solve()

    def get_pddl_state(self,gameState:GameState) -> Tuple[List[Tuple],List[Tuple]]:
        """
        This function collects pddl :objects and :init states from simulator gameState.
        """
        # Collect objects and states from the gameState

        states = []
        objects = []


        # Collect available foods on the map
        foodLeft = self.getFood(gameState).asList()
        if len(foodLeft) > 0:
            states.append(("food_available",))
        myPos = gameState.getAgentPosition(self.index)
        myObj = "a{}".format(self.index)
        cloestFoodDist = self.closestFood(myPos,self.getFood(gameState), gameState.getWalls())
        if cloestFoodDist != None and cloestFoodDist <=CLOSE_DISTANCE:
            states.append(("near_food",myObj))

        # Collect capsule states
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0 :
            states.append(("capsule_available",))
        for cap in capsules:
            if self.getMazeDistance(cap,myPos) <=CLOSE_DISTANCE:
                states.append(("near_capsule",myObj))
                break
        
        # Collect winning states
        currentScore = gameState.data.score
        if gameState.isOnRedTeam(self.index):
            if currentScore > 0:
                states.append(("winning",))
            if currentScore> 3:
                states.append(("winning_gt3",))
            if currentScore> 5:
                states.append(("winning_gt5",))
            if currentScore> 10:
                states.append(("winning_gt10",))
            if currentScore> 20:
                states.append(("winning_gt20",))
        else:
            if currentScore < 0:
                states.append(("winning",))
            if currentScore < -3:
                states.append(("winning_gt3",))
            if currentScore < -5:
                states.append(("winning_gt5",))
            if currentScore < -10:
                states.append(("winning_gt10",))
            if currentScore < -20:
                states.append(("winning_gt20",))

        # Collect team agents states
        agents : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getTeam(gameState)]
        for agent_index, agent_state in agents :
            agent_object = "a{}".format(agent_index)
            agent_type = "current_agent" if agent_index == self.index else "ally"
            objects += [(agent_object, agent_type)]

            if agent_index != self.index and self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(agent_index)) <= CLOSE_DISTANCE:
                states.append(("near_ally",))
            
            if agent_state.scaredTimer>0:
                states.append(("is_scared",agent_object))

            if agent_state.numCarrying>0:
                states.append(("food_in_backpack",agent_object))
                if agent_state.numCarrying >=20 :
                    states.append(("20_food_in_backpack",agent_object))
                if agent_state.numCarrying >=17 :
                    states.append(("17_food_in_backpack",agent_object))
                if agent_state.numCarrying >=15 :
                    states.append(("15_food_in_backpack",agent_object))
                if agent_state.numCarrying >=13 :
                    states.append(("13_food_in_backpack",agent_object))
                if agent_state.numCarrying >=10 :
                    states.append(("10_food_in_backpack",agent_object))
                # Added 1 more state for 7 food in backpack, so not too aggressive
                if agent_state.numCarrying >=7 :
                    states.append(("7_food_in_backpack",agent_object))
                if agent_state.numCarrying >=5 :
                    states.append(("5_food_in_backpack",agent_object))
                if agent_state.numCarrying >=3 :
                    states.append(("3_food_in_backpack",agent_object))
                
            if agent_state.isPacman:
                states.append(("is_pacman",agent_object))
            
            

        # Collect enemy agents states
        enemies : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getOpponents(gameState)]
        noisyDistance = gameState.getAgentDistances()
        typeIndex = 1
        for enemy_index, enemy_state in enemies:
            enemy_position = enemy_state.getPosition()
            enemy_object = "e{}".format(enemy_index)
            objects += [(enemy_object, "enemy{}".format(typeIndex))]

            if enemy_state.scaredTimer>0:
                states.append(("is_scared",enemy_object))

            if enemy_position != None:
                for agent_index, agent_state in agents:
                    if self.getMazeDistance(agent_state.getPosition(), enemy_position) <= CLOSE_DISTANCE:
                        states.append(("enemy_around",enemy_object, "a{}".format(agent_index)))
            else:
                if noisyDistance[enemy_index] >=LONG_DISTANCE :
                    states.append(("enemy_long_distance",enemy_object, "a{}".format(self.index)))
                elif noisyDistance[enemy_index] >=MEDIUM_DISTANCE :
                    states.append(("enemy_medium_distance",enemy_object, "a{}".format(self.index)))
                else:
                    states.append(("enemy_short_distance",enemy_object, "a{}".format(self.index)))                                                                                                                                                                                                 


            if enemy_state.isPacman:
                states.append(("is_pacman",enemy_object))
            typeIndex += 1
            
        return objects, states
    
    def stateSatisfyCurrentPlan(self, init_state: List[Tuple],positiveGoal, negtiveGoal):
        if self.highLevelPlan is None or len(self.highLevelPlan) == 0:
            # No plan, need a new plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            return False
        
        if positiveGoal != self.currentPositiveGoalStates or negtiveGoal != self.currentNegativeGoalStates:
            return False
        
        if self.pddl_solver.matchEffect(init_state, self.highLevelPlan[self.currentActionIndex][0] ):
            # The current state match the effect of current action, current action action done, move to next action
            if self.currentActionIndex < len(self.highLevelPlan) -1 and self.pddl_solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex+1][0]):
                # Current action finished and next action is applicable
                self.currentActionIndex += 1
                self.lowLevelPlan = [] # reset low level plan
                return True
            else:
                # Current action finished, next action is not applicable or finish last action in the plan
                return False

        if self.pddl_solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex][0]):
            # Current action precondition satisfied, continue executing current action of the plan
            return True
        
        # Current action precondition not satisfied anymore, need new plan
        return False
    
    def getGoals(self, objects: List[Tuple], initState: List[Tuple]):
        """
        My strategy is to be offensive, but in the meanwhile very cautious. I realized getting eaten is very bad.
        However, I don't like to be defensive, since best defense is to offense. 
        """
        # myObj 0 is "a0", myObj 1 is "a1". Seperate 2 agents
        myObj = "a{}".format(self.index)
        
        # First priority: Retreat if in danger and is pacman
        # Let's be very cautious and retreat if any enemy is nearby
        # Positive goal: None
        # Negative goal: Not be pacman which is go home
        if ("is_pacman", myObj) in initState:
            # Check if any enemy is around
            for obj in objects:
                agent_obj, agent_type = obj
                if agent_type in ["enemy1", "enemy2"]:
                    if ("enemy_around", agent_obj, myObj) in initState:
                        return [], [("is_pacman", myObj)]  # Goal is to not be pacman

        # Second priority: Attack if we have food and not in danger
        if ("food_in_backpack", myObj) in initState and ("is_pacman", myObj) in initState:
            # If we have nearby food and haven't collected 15 yet, try to get it
            if ("near_food", myObj) in initState and not ("15_food_in_backpack", myObj) in initState:
                # Positive goal: Have 15 food in backpack
                # Negative goal: None
                return [("15_food_in_backpack", myObj)], []
            else:
                # If no nearby food or already have 12+, go home
                # Positive goal: None
                # Negative goal: Not be pacman which is go home
                return [], [("is_pacman", myObj)]
        
        # Third priority: Eat capsule if available and enemies are nearby
        # Also I seperated the agents to have different tasks
        if ("capsule_available",) in initState and ("near_capsule", myObj) in initState:
            enemies_scared = False
            enemies_nearby = False
            for obj in objects:
                agent_obj, agent_type = obj
                if agent_type in ["enemy1", "enemy2"]:
                    if ("is_scared", agent_obj) in initState:
                        enemies_scared = True
                    # Define enemies as nearby
                    if ("enemy_around", agent_obj, myObj) in initState or \
                    ("enemy_medium_distance", agent_obj, myObj) in initState:
                        enemies_nearby = True
            
            if not enemies_scared and (enemies_nearby or len(self.getFood(self.getCurrentObservation()).asList()) > 0):
                # Find ally object
                ally_obj = None
                for obj in objects:
                    if obj[1] == "ally":
                        ally_obj = obj[0]
                        break
                # Positive goal: Eat enemy for current agent and ally go to eat food        
                if ally_obj:
                    return [("eat_enemy", myObj), ("eat_food", ally_obj)], \
                        [("capsule_available",)]        

        # Default priority: If food is available, try to collect it
        if ("food_available",) in initState:
            return [], [("food_available",)]
        else:
            # If no food available, just try to clear enemy pacmen
            negtiveGoal = []
            for obj in objects:
                agent_obj = obj[0]
                agent_type = obj[1]
                if agent_type in ["enemy1", "enemy2"] and ("is_pacman", agent_obj) in initState:
                    negtiveGoal.append(("is_pacman", agent_obj))
            if negtiveGoal:
                return [], negtiveGoal
            # If no goals possible, just return empty goals to make agent patrol
            return [], []


    #------------------------------- Heuristic search low level plan Functions -------------------------------
    def getLowLevelPlanHS(self, gameState: GameState, highLevelAction: str) -> List[Tuple[str,Tuple]]:
        # This is a function for plan low level actions using heuristic search.
        # You need to implement this function if you want to solve low level actions using heuristic search.
        # Here, we list some function you might need, read the GameState and CaptureAgent code for more useful functions.
        # These functions also useful for collecting features for Q learnning low levels.

        """
        Implement heuristic search to generate low level plan
        """
        # Get the current state of the agent and walls
        myPos = gameState.getAgentPosition(self.index)
        walls = gameState.getWalls()
        
        # Different strategies based on high level action
        # Low level plans for attack
        if highLevelAction == "attack":
                # Get all food positions
                food = self.getFood(gameState).asList()
                if not food:
                    return []

                # Get teammate info, so we can divide them into two regions and then attack
                teammates = [i for i in self.getTeam(gameState) if i != self.index]
                if teammates:  # If we have a teammate
                    teammate_index = teammates[0]
                    
                    # Divide food into upper and lower regions
                    mid_y = walls.height // 2
                    upper_food = [f for f in food if f[1] > mid_y]
                    lower_food = [f for f in food if f[1] <= mid_y]
                    
                    # Higher index agent takes upper region, lower index takes lower region
                    if self.index > teammate_index:
                        target_food = upper_food if upper_food else lower_food  # Fallback if empty
                    else:
                        target_food = lower_food if lower_food else upper_food
                        
                    # Find the safest food in our assigned region
                    target = self.findSafestFood(gameState, target_food)
                else:
                    # If no teammate, just find the safest food overall
                    target = self.findSafestFood(gameState, food)
                    
                if not target:
                    return []
                    
                path = self.aStarSearch(gameState, myPos, target, 
                                    lambda x, y: self.attackHeuristic(x, y, gameState))
            
        elif highLevelAction == "collect_nearby_food":
            food = self.getFood(gameState)
            nearby_food = []
            # Define nearby as within 5 steps
            for x in range(walls.width):
                for y in range(walls.height):
                    if food[x][y] and self.getMazeDistance(myPos, (x,y)) <= 5:
                        nearby_food.append((x,y))
                        
            if nearby_food:
                # Find closest nearby food
                target = min(nearby_food, key=lambda x: self.getMazeDistance(myPos, x))
                path = self.aStarSearch(gameState, myPos, target,
                                    lambda x, y: self.getMazeDistance(x, y))
                return self.pathToActions(path)

        elif highLevelAction == "go_home":
            # Find closest position in our territory
            mid = walls.width // 2
            if self.red:
                homePosY = [y for y in range(walls.height) if not walls[mid-1][y]]
                homePos = [(mid-1, y) for y in homePosY]
            else:
                homePosY = [y for y in range(walls.height) if not walls[mid][y]]
                homePos = [(mid, y) for y in homePosY]
                
            target = min(homePos, key=lambda x: self.getMazeDistance(myPos, x))
            path = self.aStarSearch(gameState, myPos, target,
                                lambda x, y: self.escapeHeuristic(x, y, gameState))
            
        # Handle capsule eating and becoming Pacman
        elif highLevelAction == "eat_capsule":
            capsules = self.getCapsules(gameState)
            if capsules:
                # Find closest capsule
                target = min(capsules, key=lambda x: self.getMazeDistance(myPos, x))
                path = self.aStarSearch(gameState, myPos, target,
                                    lambda x, y: self.capsuleHeuristic(x, y, gameState))
                return self.pathToActions(path)
            
        else:
            # Patrol if no high level action
            target = self.startPosition
            path = self.aStarSearch(gameState, myPos, target,
                                lambda x, y: self.getMazeDistance(x, y))


        # Convert path to actions
        return self.pathToActions(path)

    def attackHeuristic(self, pos, goal, gameState):
        """Heuristic for attacking that considers ghosts and score"""
        # Base distance
        distance = self.getMazeDistance(pos, goal)
        
        # Ghost avoidance
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        
        # High penalty for being near a ghost
        for ghost in ghosts:
            ghostPos = ghost.getPosition()
            ghostDist = self.getMazeDistance(pos, ghostPos)
            if ghostDist < 2:
                distance += 200
            elif ghostDist < 4:
                distance += 50
                
        return distance

    def defenseHeuristic(self, pos, goal, gameState):
        """Heuristic for defense that prioritizes intercepting paths"""
        # Base distance to target
        distance = self.getMazeDistance(pos, goal)
        
        # Get valid positions in homeLine
        mid = gameState.getWalls().width // 2
        walls = gameState.getWalls()
        homeLine = []

        # Populate valid home line positions based on team color
        if self.red:
            homeLine = [(mid - 1, y) for y in range(walls.height) if not walls[mid - 1][y]]
        else:
            homeLine = [(mid, y) for y in range(walls.height) if not walls[mid][y]]

        # Ensure positions are within grid and calculate minimum distance
        minHomeDistance = min([self.getMazeDistance(pos, home) for home in homeLine if home in gameState.getWalls().asList(False)])
        
        return distance + minHomeDistance * 0.5

    def escapeHeuristic(self, pos, goal, gameState):
        """Heuristic for escaping that heavily weights ghost avoidance"""
        distance = self.getMazeDistance(pos, goal)
        
        # Ghost avoidance with higher penalties
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        
        for ghost in ghosts:
            ghostPos = ghost.getPosition()
            if ghostPos:
                ghostDist = self.getMazeDistance(pos, ghostPos)
                if ghostDist < 3:
                    distance += 200
                elif ghostDist < 5:
                    distance += 100
                    
        return distance
    def capsuleHeuristic(self, pos, goal, gameState):
        """Heuristic for capsule eating that considers ghost positions"""
        distance = self.getMazeDistance(pos, goal)
        
        # Ghost avoidance with different weights for scared vs non-scared ghosts
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        
        for ghost in ghosts:
            ghostPos = ghost.getPosition()
            if ghostPos:
                ghostDist = self.getMazeDistance(pos, ghostPos)
                # Less penalty for scared ghosts
                if ghost.scaredTimer > 0:
                    if ghostDist < 2:
                        distance += 10
                else:
                    if ghostDist < 2:
                        distance += 100
                    elif ghostDist < 4:
                        distance += 50
                        
        return distance

    def aStarSearch(self, gameState, start, goal, heuristic):
        """A* search implementation"""
        priority_queue = util.PriorityQueue()  # Priority queue 
        priority_queue.push((start, []), 0)  # Push start position with empty path
        visited = set()  # Set to store visited positions
        
        while not priority_queue.isEmpty():
            pos, path = priority_queue.pop()
            
            if pos == goal:
                return path + [pos]
                
            if pos in visited:
                continue
                
            visited.add(pos)
            
            # Get legal successors
            successors = Actions.getLegalNeighbors(pos, gameState.getWalls())
            for nextPos in successors:
                if nextPos not in visited:
                    newPath = path + [pos]
                    priority = len(newPath) + heuristic(nextPos, goal)
                    priority_queue.push((nextPos, newPath), priority)
        
        return []

    def pathToActions(self, path):
        """Convert path positions to action tuples"""
        if not path or len(path) < 2:
            return [("Stop", path[0])] if path else []
            
        actions = []
        for i in range(len(path) - 1):
            curr = path[i]
            next = path[i + 1]
            dx = next[0] - curr[0]
            dy = next[1] - curr[1]
            
            if dx == 1:
                actions.append(("East", next))
            elif dx == -1:
                actions.append(("West", next))
            elif dy == 1:
                actions.append(("North", next))
            elif dy == -1:
                actions.append(("South", next))
                
        return actions

    def findSafestFood(self, gameState, foodList):
        """Find the safest food to target considering ghosts and distance"""
        myPos = gameState.getAgentPosition(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        
        # If no ghosts visible, just find closest food
        if not ghosts:
            return min(foodList, key=lambda x: self.getMazeDistance(myPos, x))
            
        # Score each food based on distance and ghost proximity
        foodScores = {}
        for food in foodList:
            score = self.getMazeDistance(myPos, food)
            for ghost in ghosts:
                ghostPos = ghost.getPosition()
                if ghostPos:
                    ghostDist = self.getMazeDistance(food, ghostPos)
                    if ghostDist < 3:
                        score += 100
                    elif ghostDist < 5:
                        score += 50
            foodScores[food] = score
            
        return min(foodScores.keys(), key=lambda x: foodScores[x])
    
    
    def posSatisfyLowLevelPlan(self,gameState: GameState):
        if self.lowLevelPlan == None or len(self.lowLevelPlan)==0 or self.lowLevelActionIndex >= len(self.lowLevelPlan):
            return False
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,self.lowLevelPlan[self.lowLevelActionIndex][0])
        if nextPos != self.lowLevelPlan[self.lowLevelActionIndex][1]:
            return False
        return True

    #------------------------------- Q-learning low level plan Functions -------------------------------

    """
    Iterate through all q-values that we get from all
    possible actions, and return the action associated
    with the highest q-value.
    """
    def getLowLevelPlanQL(self, gameState:GameState, highLevelAction: str) -> List[Tuple[str,Tuple]]:
        values = []
        legalActions = gameState.getLegalActions(self.index)
        rewardFunction = None
        featureFunction = None
        weights = None
        learningRate = 0

        ##########
        # The following classification of high level actions is only a example.
        # You should think and use your own way to design low level planner.
        ##########
        if highLevelAction == "attack":
            # The q learning process for offensive actions are complete, 
            # you can improve getOffensiveFeatures to collect more useful feature to pass more information to Q learning model
            # you can improve the getOffensiveReward function to give reward for new features and improve the trainning process .
            rewardFunction = self.getOffensiveReward
            featureFunction = self.getOffensiveFeatures
            weights = self.getOffensiveWeights()
            learningRate = self.alpha
        elif highLevelAction == "go_home":
            # The q learning process for escape actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getEscapeReward
            featureFunction = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            learningRate = 0 # learning rate set to 0 as reward function not implemented for this action, do not do q update, 
        else:
            # The q learning process for defensive actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getDefensiveReward
            featureFunction = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            learningRate = 0 # learning rate set to 0 as reward function not implemented for this action, do not do q update 

        if len(legalActions) != 0:
            prob = util.flipCoin(self.epsilon) # get change of perform random movement
            if prob and self.trainning:
                action = random.choice(legalActions)
            else:
                for action in legalActions:
                        if self.trainning:
                            self.updateWeights(gameState, action, rewardFunction, featureFunction, weights,learningRate)
                        values.append((self.getQValue(featureFunction(gameState, action), weights), action))
                action = max(values)[1]
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,action)
        return [(action, nextPos)]


    """
    Iterate through all features (closest food, bias, ghost dist),
    multiply each of the features' value to the feature's weight,
    and return the sum of all these values to get the q-value.
    """
    def getQValue(self, features, weights):
        return features * weights
    
    """
    Iterate through all features and for each feature, update
    its weight values using the following formula:
    w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
    """
    def updateWeights(self, gameState, action, rewardFunction, featureFunction, weights, learningRate):
        features = featureFunction(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        reward = rewardFunction(gameState, nextState)
        for feature in features:
            correction = (reward + self.discountRate*self.getValue(nextState, featureFunction, weights)) - self.getQValue(features, weights)
            weights[feature] =weights[feature] + learningRate*correction * features[feature]
        
    
    """
    Iterate through all q-values that we get from all
    possible actions, and return the highest q-value
    """
    def getValue(self, nextState: GameState, featureFunction, weights):
        qVals = []
        legalActions = nextState.getLegalActions(self.index)

        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                features = featureFunction(nextState, action)
                qVals.append(self.getQValue(features,weights))
            return max(qVals)
    
    def getOffensiveReward(self, gameState: GameState, nextState: GameState):
        # Calculate the reward. 
        currentAgentState:AgentState = gameState.getAgentState(self.index)
        nextAgentState:AgentState = nextState.getAgentState(self.index)

        ghosts = self.getGhostLocs(gameState)
        ghost_1_step = sum(nextAgentState.getPosition() in Actions.getLegalNeighbors(g,gameState.getWalls()) for g in ghosts)

        base_reward =  -50 + nextAgentState.numReturned + nextAgentState.numCarrying
        new_food_returned = nextAgentState.numReturned - currentAgentState.numReturned
        score = self.getScore(nextState)

        if ghost_1_step > 0:
            base_reward -= 5
        if score <0:
            base_reward += score
        if new_food_returned > 0:
            # return home with food get reward score
            base_reward += new_food_returned*10
        
        print("Agent ", self.index," reward ",base_reward)
        return base_reward
    
    def getDefensiveReward(self,gameState, nextState):
        print("Warnning: DefensiveReward not implemented yet, and learnning rate is 0 for defensive ",file=sys.stderr)
        return 0
    
    def getEscapeReward(self,gameState, nextState):
        print("Warnning: EscapeReward not implemented yet, and learnning rate is 0 for escape",file=sys.stderr)
        return 0



    #------------------------------- Feature Related Action Functions -------------------------------


    
    def getOffensiveFeatures(self, gameState: GameState, action):
        food = self.getFood(gameState) 
        currAgentState = gameState.getAgentState(self.index)

        walls = gameState.getWalls()
        ghosts = self.getGhostLocs(gameState)
        
        # Initialize features
        features = util.Counter()
        nextState = self.getSuccessor(gameState, action)

        # Successor Score
        features['successorScore'] = self.getScore(nextState)/(walls.width+walls.height) * 10

        # Bias
        features["bias"] = 1.0
        
        # Get the location of pacman after he takes the action
        next_x, next_y = nextState.getAgentPosition(self.index)

        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts) 
        
        
        dist_home =  self.getMazeDistance((next_x, next_y), gameState.getInitialAgentPosition(self.index))+1

        features["chance-return-food"] = (currAgentState.numCarrying)*(1 - dist_home/(walls.width+walls.height)) # The closer to home, the larger food carried, more chance return food
        
        # Closest food
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features["closest-food"] = dist/(walls.width+walls.height)
        else:
            features["closest-food"] = 0

        return features

    def getOffensiveWeights(self):
        return MixedAgent.QLWeights["offensiveWeights"]
    


    def getEscapeFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemiesAround = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(enemiesAround) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemiesAround]
            features['enemyDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        features["distanceToHome"] = self.getMazeDistance(myPos,self.startPosition)

        return features

    def getEscapeWeights(self):
        return MixedAgent.QLWeights["escapeWeights"]
    


    def getDefensiveFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        team = [successor.getAgentState(i) for i in self.getTeam(successor)]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())
        features['teamDistance'] = team_dist

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getDefensiveWeights(self):
        return MixedAgent.QLWeights["defensiveWeights"]
    
    def closestFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None
    
    def stateClosestFood(self, gameState:GameState):
        pos = gameState.getAgentPosition(self.index)
        food = self.getFood(gameState)
        walls = gameState.getWalls()
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None
    
    def getSuccessor(self, gameState: GameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
    
    def getGhostLocs(self, gameState:GameState):
        ghosts = []
        opAgents = CaptureAgent.getOpponents(self, gameState)
        # Get ghost locations and states if observable
        if opAgents:
                for opponent in opAgents:
                        opPos = gameState.getAgentPosition(opponent)
                        opIsPacman = gameState.getAgentState(opponent).isPacman
                        if opPos and not opIsPacman: 
                                ghosts.append(opPos)
        return ghosts
    

