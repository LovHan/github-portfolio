# heuristics/gridmap_h.py
#
# Heuristics for gridmap.
#
# @author: mike
# @created: 2020-07-22
#

import math, random
from lib_piglet.search.search_node import compare_node_g, compare_node_f, search_node
from lib_piglet.utils.data_structure import bin_heap

def piglet_heuristic(domain,current_state, goal_state):
    return straight_heuristic(current_state, goal_state)

def pigelet_multi_agent_heuristic(domain,current_state, goal_state):
    h = 0
    for agent, loc in current_state.agent_locations_.items():
        h += manhattan_heuristic(loc, goal_state.agent_locations_[agent])
    return h

def manhattan_heuristic(current_state, goal_state):
    return abs(current_state[0] - goal_state[0]) + abs(current_state[1] - goal_state[1])


def straight_heuristic(current_state, goal_state):
    return round(math.sqrt((current_state[0] - goal_state[0])**2 + (current_state[1] - goal_state[1])**2), 5)

def octile_heuristic(current_state, goal_state):
    delta_x = abs(current_state[0] - goal_state[0])
    delta_y = abs(current_state[1] - goal_state[1])
    return min(delta_x, delta_y) * math.sqrt(2) + max(delta_x,delta_y) - min(delta_x, delta_y)


pivots = {}


def differential_heuristic(domain, current_state,goal_state):
    if len(pivots) == 0:
        while len(pivots) < 5:
            random_loc = (random.randint(0,domain.height_), random.randint(0,domain.width_))
            if domain.get_tile(random_loc):
                pivots[random_loc] = {}
        # self.pivots = {(97,6):{}}
        for pivot in pivots.keys():
            pivots[pivot] = calculate_distance(domain, pivot)
    
    all_h = []
    for pivot in pivots.keys(): 
        all_h.append( abs(pivots[pivot][current_state] - pivots[pivot][goal_state]) )
    return max(all_h)

def true_dis_heuristic(domain, current_state,goal_state):
    if goal_state not in pivots:
        pivots[goal_state] = calculate_distance(domain, goal_state)
    return pivots[goal_state][current_state]


def calculate_distance(domain,pivot):
    open = bin_heap(compare_node_g)
    all_nodes = {}
    start = search_node()
    start.state_ = pivot
    start.g_ = 0
    start.priority_queue_handle_ = open.push(start)
    all_nodes[start] = start
    distance_table = {start.state_: 0}
    while len(open) >0:
        current = open.pop()
        current.close()
        distance_table[current.state_] = current.g_


        for action in [(1,0,1), (-1,0,1), (0,1,1),(0,-1,1), (-1,-1,1.41), (-1,+1,1.41), (+1,+1,1.41), (+1,-1,1.41)]:
            succ_state = (current.state_[0] + action[0], current.state_[1] + action[1])
            if not domain.get_tile(succ_state):
                continue
            succ_node = search_node()
            succ_node.state_ = succ_state
            succ_node.g_ = current.g_ + action[2]
            
            if succ_node not in all_nodes:
                # we need this open_handle_ to update the node in open list in the future
                succ_node.priority_queue_handle_ = open.push(succ_node)
                all_nodes[succ_node] = succ_node
            else:
                # succ_node only have the same hash and state comparing with the on in the all nodes list
                # It's not the one in the all nodes list,  we need the real node in the all nodes list.
                exist = all_nodes[succ_node]
                if not exist.is_closed() and exist.g_ > succ_node.g_:
                    exist.g_ = succ_node.g_
                    exist.parent_ = succ_node.parent_
                    if exist.open_handle_ is not None:
                        # If handle exist, we are using bin_heap. We need to tell bin_heap one element's value
                        # is decreased. Bin_heap will update the heap to maintain priority structure.
                        open.decrease(exist.open_handle_)
                
    return distance_table