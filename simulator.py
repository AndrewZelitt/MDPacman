import nodes
import constants
import random
from collections import deque
import pygame_widgets
import pygame
from mdp_value_iteration import ValueIterationPacmanMDP

def run_pacman_simulation(self, game_map, pacman_start, ghost_starts, pellet_positions, max_moves=10000):
    """
    Run a Pacman simulation and return the final score.
    
    Uses MDP-based value iteration to determine optimal Pacman moves.
    Ghosts actively chase Pacman using pathfinding.
    
    Args:
        game_map: the nodegroup class object from the pacman code
        pacman_start: tuple (x, y) starting position
        ghost_starts: list of ghosts for starting positions
        pellet_positions: list of pellets
        max_moves: maximum number of moves before game ends
    
    Returns:
        int: final score
    """
    # Initialize Value-Iteration MDP solver for pacman-only abstraction
    mdp_solver = ValueIterationPacmanMDP(game_map)
    
    # Game state
    score = 0
    power_mode = 0  # Turns remaining in power mode
    moves = 0

    # Configure Pacman's start position
    try:
        if pacman_start is None or pacman_start == 'random':
            candidates = list(game_map.nodesLUT.values())
            start_node = random.choice(candidates)
            # set pacman's start node and position
            try:
                self.pacman.setStartNode(start_node)
            except Exception:
                self.pacman.node = start_node
                self.pacman.setPosition()
        elif isinstance(pacman_start, tuple) or isinstance(pacman_start, list):
            # pacman_start expected as (x,y) tile coords (may be floats)
            x, y = int(pacman_start[0]), int(pacman_start[1])
            try:
                start_node = game_map.getNodeFromTiles(x, y)
                self.pacman.setStartNode(start_node)
            except Exception:
                # fallback: leave current pacman position
                pass
    except Exception:
        # if anything goes wrong, continue with existing pacman position
        pass
    
    while moves < max_moves and len(self.pellets.pelletList) > 0:
        # Pacman collects pellets at current location
        pellets_to_remove = []
        for pellet in self.pellets.pelletList:
            if self.pacman.node.coords == pellet.coord:
                if pellet.name == constants.POWERPELLET:
                    power_mode = 20
                    score += 50
                else:
                    score += 10
                pellets_to_remove.append(pellet)
        
        for pellet in pellets_to_remove:
            self.pellets.pelletList.remove(pellet)

        # Check ghost collisions
        for ghost in self.ghosts:
            if self.pacman.node.coords == ghost.node.coords:
                if power_mode > 0:
                    score += 200  # Eat ghost
                    ghost.node = self.nodes.getNodeFromTiles(2+11.5, 3+14)
                    power_mode -= 1
                else:
                    return score
        
        # Move Pacman using MDP policy
        move_pacman(self, game_map, mdp_solver, power_mode)

        # Move ghosts (chase Pacman or flee in power mode)
        for ghost in self.ghosts:
            move_ghost(self, game_map, ghost, self.pacman.node, power_mode > 0)

        # Decrease power mode
        if power_mode > 0:
            power_mode -= 1
        
        moves += 1

        # Bonus for completing level
        if len(self.pellets.pelletList) == 0:
            score += 1000
            print("Completed Level")
            return score

    return score

def move_pacman(self, game_map, mdp_solver, power_mode):
    """
    Move Pacman using MDP-based policy.
    
    Args:
        self: Game controller instance
        game_map: NodeGroup object
        mdp_solver: SimplePacmanMDPSolver instance
        power_mode: Current power mode turns remaining
    """
    # Get the best action using MDP policy
    best_action = mdp_solver.get_next_move(
        self.pacman.node, 
        self.pellets.pelletList, 
        self.ghosts, 
        power_mode
    )
    
    # Move to the next node in the direction recommended by policy
    next_node = self.pacman.node.neighbors.get(best_action)
    
    if next_node is not None:
        self.pacman.node = next_node
    
    return self.pacman.node


def next_step(start, goal, game_map):
    """this is a helper function that finds the next position for the ghosts to be in"""
    if start == goal:
        return start, 0
    queue = deque([(start, [start])])  # Stores (current_node, path_to_current_node)
    visited = {start}  # Keeps track of visited nodes to avoid cycles

    while queue:
        current_node, path = queue.popleft()
        
        if current_node == goal:
            #print("found the path", path[1])
            return path[1], len(path)
        #print(current_node.coords)
        #this is amoung the dumbest code I have ever written imo
        for nod in game_map.nodesLUT:
            if current_node.coords == game_map.nodesLUT[nod].coords:
                for neighbor in game_map.nodesLUT[nod].neighbors.values():  
                    if neighbor is not None:
                        if neighbor not in visited:
                            #print(neighbor.coords)
                            visited.add(neighbor)
                            queue.append((neighbor, path + [neighbor]))


    #Zakhar algo
    """queue = deque([start])
    visited = {start: None}

    while queue:
        current = queue.popleft()
        for node in game_map.nodesLUT.keys():
            for neighbor in game_map.nodesLUT[node].neighbors.values():
                #if neighbor is not None:
                    #print(node, "has neighbor: ", neighbor.coords)
                    if neighbor not in visited:
                        visited[neighbor] = current
                        queue.append(neighbor)

                        if neighbor == goal:
                            print("found goal", start.coords, goal.coords)
                            for p in visited:
                                if p is not None:
                                    print(p.coords)
                            path = [neighbor]
                            while visited[path[-1]] is not None:
                                print("path[-1] == ", path[-1].coords)
                                path.append(visited[path[-1]])
                            path.reverse()

                            print("Path = ")
                            for p in path:
                                print(p.coords)

                            if len(path) > 1:
                                return path[1]
                            return start"""

    return start, 1  # failsafe but like this sohuld never happen


def move_ghost(self, game_map, ghost, pacman_node, flee_mode):
    """Move ghost toward (or away from) Pacman using pathfinding."""
    
    if flee_mode:
        # Run away from Pacman - move to neighboring node farthest from Pacman
        best_node = ghost.node
        best_dist = -1

        for neighbor in ghost.node.neighbors.values():
            if neighbor is not None:
                dist = abs(neighbor.coords[0] - pacman_node.coords[0]) + abs(neighbor.coords[1] - pacman_node.coords[1])
                if dist > best_dist:
                    best_dist = dist
                    best_node = neighbor
        
        ghost.node = best_node
    else:
        # Chase Pacman - find shortest path and move one step closer
        next_node, num_steps = next_step(ghost.node, pacman_node, game_map)
        ghost.node = next_node