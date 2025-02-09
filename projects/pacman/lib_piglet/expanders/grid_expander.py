# expander/grid_expander.py
# 
# Expand function for the 4-connected gridmap domain.
#
# Given a current search node, the expander checks the set of valid grid actions 
# and generates search node successors for each.
#
# @author: dharabor
# @created: 2020-07-15
#

from lib_piglet.search.search_node import search_node
from lib_piglet.expanders.base_expander import base_expander
from lib_piglet.domains.gridmap import gridmap, gridmap_joint, grid_joint_state
from lib_piglet.domains.grid_action import  Move_Actions, grid_action
from lib_piglet.constraints.grid_constraints import grid_constraint_table, grid_reservation_table
import copy

class grid_expander(base_expander):


    def __init__(self, map : gridmap, constraint_table: grid_constraint_table = None):
        self.domain_: gridmap = map
        self.effects_: list = [self.domain_.height_*-1, self.domain_.height_, -1, 1]
        self.constraint_table_: grid_constraint_table   = constraint_table
        self.reservation_table_: grid_reservation_table = None # reservation_table_ is not used on default, decide how to use it on your own.

        # memory for storing successor (state, action) pairs
        self.succ_: list = [None] * 4 

        # pre-allocate a pool of search nodes
        self.nodes_: list = []
        for x in range(self.domain_.height_ * self.domain_.width_):
            self.nodes_.append(search_node())

    # identify successors of the current node
    #
    # @param current: The current node
    # @return : Possible next
    def expand(self, current: search_node):
        
        self.succ_.clear()
        for a in self.get_actions(current.state_):
            # NB: we only initialise the state and action attributes.
            # The search will initialise the rest, assuming it decides 
            # to add the corresponding successor to OPEN
            new_state = self.__move(current.state_, a.move_)
            if self.constraint_table_ is not None:
                if self.constraint_table_.get_constraint(new_state,current.timestep_+1).v_:
                    continue
                if self.constraint_table_.get_constraint(current.state_,current.timestep_).e_[a.move_]:
                    continue
            self.succ_.append((new_state, a))
        return self.succ_[:]

        # return a list with all the applicable/valid actions
        # at tile (x, y)
        # @param loc A (x,y) coordinate tuple
        # @return a list of gridaction object.

    def get_actions(self, loc: tuple):
        x = loc[0]
        y = loc[1]
        retval = []

        if (x < 0 or x >= int(self.domain_.height_) or y < 0 or y >= int(self.domain_.width_)):
            return retval

        if (self.domain_.get_tile(loc) == False):
            return retval

        if (self.domain_.get_tile((x,y - 1))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_LEFT
            retval[-1].cost_ = 1;

        if (self.domain_.get_tile((x,y + 1))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_RIGHT
            retval[-1].cost_ = 1;

        if (self.domain_.get_tile((x - 1,y))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_UP
            retval[-1].cost_ = 1;

        if (self.domain_.get_tile((x + 1,y))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_DOWN
            retval[-1].cost_ = 1;

        if (self.domain_.get_tile((x - 1,y - 1))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_UP_LEFT
            retval[-1].cost_ = 1.41;
        if (self.domain_.get_tile((x - 1,y + 1))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_UP_RIGHT
            retval[-1].cost_ = 1.41;
        if (self.domain_.get_tile((x + 1,y + 1))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_DOWN_RIGHT
            retval[-1].cost_ = 1.41;
        if (self.domain_.get_tile((x + 1,y - 1))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_DOWN_LEFT
            retval[-1].cost_ = 1.41;

        return retval

    def __move(self, curr_state: tuple, move):
        x = curr_state[0]
        y = curr_state[1]
        if move == Move_Actions.MOVE_UP:
            x -= 1
        elif move == Move_Actions.MOVE_DOWN:
            x += 1
        elif move == Move_Actions.MOVE_LEFT:
            y -= 1
        elif move == Move_Actions.MOVE_RIGHT:
            y += 1
        elif move == Move_Actions.MOVE_UP_LEFT:
            x -= 1
            y -= 1
        elif move == Move_Actions.MOVE_UP_RIGHT:
            x -= 1
            y += 1
        elif move == Move_Actions.MOVE_DOWN_RIGHT:
            x += 1
            y += 1
        elif move == Move_Actions.MOVE_DOWN_LEFT:
            x += 1
            y -= 1
        return x, y

    def __str__(self):
        return str(self.domain_)




class grid_joint_expander(base_expander):


    def __init__(self, map: gridmap, constraint_table: grid_constraint_table = None):
        self.domain_: gridmap_joint = map
        self.effects_: list = [self.domain_.height_ * -1, self.domain_.height_, -1, 1]
        self.constraint_table_: grid_constraint_table  = constraint_table
        self.reservation_table_: grid_reservation_table = None   # reservation_table_ is not used on default, decide how to use it on your own.

        # memory for storing successor (state, action) pairs
        self.succ_: list = [None] * 4

        # pre-allocate a pool of search nodes
        self.nodes_: list = []
        for x in range(self.domain_.height_ * self.domain_.width_):
            self.nodes_.append(search_node())

    # identify successors of the current node
    #
    # @param current: The current node
    # @return : Possible next
    def expand(self, current: search_node):

        self.succ_.clear()
        current_state : grid_joint_state =  copy.deepcopy(current.state_)
        # for key, item in current.state_.agent_locations_.items():
        #     if self.domain_.goal_.agent_locations_[key] == item:
        #         current_state.agent_locations_.pop(key)

        agents_left = list(current_state.agent_locations_.keys())
        loc_set = {}
        self.generate_states(current_state, agents_left,len(current_state.agent_locations_), loc_set, current_state)

        return self.succ_[:]

    def generate_states(self, current_state: grid_joint_state, agents_left: list, cost: int, loc_set, parent_state):
        if len(agents_left) == 0:
            action = grid_action()
            action.move_ = None
            action.cost_ = cost
            self.succ_.append((current_state, action))
            return

        agent = agents_left.pop(0)
        loc = current_state.agent_locations_[agent]

        for a in self.get_actions(loc):
            # NB: we only initialise the state and action attributes.
            # The search will initialise the rest, assuming it decides
            # to add the corresponding successor to OPEN
            new_loc = self.__move(loc, a.move_)
            if new_loc in loc_set:
                continue
            if loc in loc_set and parent_state.agent_locations_[loc_set[loc]] == new_loc:
                continue
            new_set = copy.deepcopy(loc_set)
            new_set[new_loc]=agent
            new_state = copy.deepcopy(current_state)
            new_state.agent_locations_[agent] = new_loc
            self.generate_states(new_state, agents_left[:],cost,new_set,parent_state)


    # return a list with all the applicable/valid actions
    # at tile (x, y)
    # @param loc A (x,y) coordinate tuple
    # @return a list of gridaction object.
    def get_actions(self, loc: tuple):
        x = loc[0]
        y = loc[1]
        retval = []

        if (x < 0 or x >= int(self.domain_.height_) or y < 0 or y >= int(self.domain_.width_)):
            return retval

        if (self.domain_.get_tile(loc) == False):
            return retval

        if (self.domain_.get_tile((x, y - 1))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_LEFT
            retval[-1].cost_ = 1;

        if (self.domain_.get_tile((x, y + 1))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_RIGHT
            retval[-1].cost_ = 1;

        if (self.domain_.get_tile((x - 1, y))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_UP
            retval[-1].cost_ = 1;

        if (self.domain_.get_tile((x + 1, y))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_DOWN
            retval[-1].cost_ = 1;

        if (self.domain_.get_tile((x, y))):
            retval.append(grid_action())
            retval[-1].move_ = Move_Actions.MOVE_WAIT
            retval[-1].cost_ = 1;

        return retval

    def __move(self, curr_state: tuple, move):
        x = curr_state[0]
        y = curr_state[1]
        if move == Move_Actions.MOVE_UP:
            x -= 1
        elif move == Move_Actions.MOVE_DOWN:
            x += 1
        elif move == Move_Actions.MOVE_LEFT:
            y -= 1
        elif move == Move_Actions.MOVE_RIGHT:
            y += 1

        return x, y

    def __str__(self):
        return str(self.domain_)