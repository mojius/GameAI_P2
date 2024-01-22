
from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

num_nodes = 250
explore_faction = 2.

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node

    """
    # traverse tree until reaches node that can be expanded (a node w/ untried actions) 
        # or until you reach a terminal node (end of the game).
    while True:
        #base cases
        if board.is_ended(state): # if node is terminal
            return node, state
        if node.untried_actions: # if node can be expanded
            return node, state
        #if node !terminal & !have untried actions, select child node w/ highest UCB score
        ucb_scores = {action: ucb(child, bot_identity != child.parent_action) for action, child in node.child_nodes.items()}
        # ^ dictionary compression, for e/a key pair val: action, child in the nodes child nodes, it calculates the UCB & adds to UCB dict -> action: score
        best_action = max(ucb_scores, key=ucb_scores.get) # rets action that has highest ucb score
        
        # move iters
        node = node.child_nodes[best_action]
        state = board.next_state(state, best_action)
    
    # once expandable node or a terminal node found, return it along with the corresponding game state.
    pass

def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
    new_action = choice(node.untried_actions) # gen random action from untried actions
    new_state = board.next_state(state, new_action) # gen state from chosen action
    new_node = MCTSNode(node, new_action, board.legal_actions(new_state)) # create new node representing new state
    node.child_nodes[new_action] = new_node # add new node to child_nodes dictionary of the parent node
    node.untried_actions.remove(new_action) # remove chosen action from the untried actions of the parent node
    
    return new_node, new_state #returns new node and the new state


def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    while not board.is_ended(state):
        action = choice(board.legal_actions(state))  #chose rand action from legal actions in curr state
        state = board.next_state(state, action)  # apply action to state
    
    return state


def backpropagate(node: MCTSNode|None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    if node is None: # base case: root node reached
        return
    
    node.visits += 1 # node has been visited
    if won:
        node.wins += 1

    return backpropagate (node.parent, won)


def ucb(node: MCTSNode, is_opponent: bool):
    """ Calcualtes the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """
    #C = sqrt(2) # common choice for exploration constant from what i've seen o.0 

    if node.visits == 0: # if node has never been visited (avoid /0 & encourage exploration of unvisited nodes)
        return float('inf')
    
    win_rate = node.wins / node.visits #calc win rate of node

    if is_opponent: #accomodates for if node is opponents
        win_rate = 1 - win_rate 
    
    total_visits_parent = node.parent.visits if node.parent is not None else 1 # just in case, don't see why parent should be none tho

    ucb_value = win_rate + explore_faction * sqrt(log(total_visits_parent) / node.visits) 
    
    return ucb_value
    pass

def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
    best_action = None
    #max_visits = -1
    max_win_rate = -1.0
    
    for action, child_node in root_node.child_nodes.items(): # iters thru e/a poss action that has a node affiliated w/ it
        #by most visited node (tested this method, doesn't get as many wins as the win rate method does)
        #if child_node.visits > max_visits:
        #    best_action = action
        #    max_visits = child_node.visits
            
        #by win rate
        win_rate = child_node.wins / child_node.visits if child_node.visits > 0 else 0 # if has been visited, calcs win rate
        if win_rate > max_win_rate: #if win rate of curr > max, update
            best_action = action
            max_win_rate = win_rate

    return best_action


def is_win(board: Board, state, identity_of_bot: int):  
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.
    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state
    """
    bot_identity = board.current_player(current_state) # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))
    
    state = current_state
    node = root_node
    for _ in range(num_nodes):
        # Do MCTS - This is all you!
        node, state = traverse_nodes(root_node, board, current_state, bot_identity)

        if node.untried_actions: #expand node if has untried actions
            node, state = expand_leaf(node, board, state)
        
        terminal_state = rollout(board, state) # rollout from the new node's state

        won = is_win(board, terminal_state, bot_identity) #if bot won rollout 
        
        backpropagate(node, won) #update tree
    
    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node)
    
    #print(f"Action chosen: {best_action}")
    return best_action
