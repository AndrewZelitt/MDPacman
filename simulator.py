import nodes
import constants
import random
from collections import deque
import pygame_widgets
import pygame

def run_pacman_simulation(self, game_map, pacman_start, ghost_starts, pellet_positions, max_moves=10000):
    """
    Run a Pacman simulation and return the final score.
    
    Args:
        game_map: the nodegroup class object from the pacman code
                
        pacman_start: tuple (x, y) starting position
        ghost_starts: list of tuples [(x, y), ...] for ghost starting positions
        max_moves: maximum number of moves before game ends
    
    Returns:
        int: final score
    """
    # Initialize game state
    pacman_pos = pacman_start
    ghost_positions = ghost_starts
    score = 0
    pellets_remaining = len(pellet_positions)
    power_mode = 0  # Turns remaining in power mode
    moves = 0
    
    while moves < max_moves and pellets_remaining > 0:
        # Pacman collects pellets
        #print(f"pacman pos: {pacman_pos}")
        #print(f"ghost pos: {ghost_positions}")
        for pellet in self.pellets.pelletList:
            if self.pacman.node.coords == pellet.coord:
                if pellet.name == constants.POWERPELLET:
                    self.pellets.pelletList.remove(pellet)
                    power_mode = 20
                    score += 50
                else:
                    score += 10
                    self.pellets.pelletList.remove(pellet)

        # Need to figure this stuff up, just doing the movement for now.
        """
        current_node = self.pacman.node.coords
        
        if current_node.get('pellet', False):
            score += 10
            current_node['pellet'] = False
            pellets_remaining -= 1
            
        if current_node.get('power_pellet', False):
            score += 50
            current_node['power_pellet'] = False
            power_mode = 20  # Power mode lasts 20 turns

        if current_node.get('fruit', False):
            score += 200
            current_node['fruit'] = False
        
        
       """ 

        # Check ghost collisions
        for ghost in ghost_positions:
            if self.pacman.node.coords == ghost.node.coords:
                if power_mode > 0:
                    score += 200  # Eat ghost
                    ghost.node = self.nodes.getNodeFromTiles(2+11.5, 3+14)
                else:
                    #print("Score = ", score)
                    return score
            
        
        # Move Pacman (simple strategy: move toward nearest pellet)
        # this will be replaced with the mdp stuff
        pacman_pos = move_pacman(self, game_map, pacman_pos, pellets_remaining)

        # Move ghosts (simple chase strategy)
        [move_ghost(self, game_map, ghost_pos, pacman_pos, power_mode > 0) for ghost_pos in ghost_positions]

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

def move_pacman(self, game_map, current_pos, pellets_remaining):
    rand = random.randrange(1, 5)

    # ^^^ this will be replaced with the output from the Policys


    
    if rand == 1:
        direction =  constants.UP
    if rand == 2:
        direction =  constants.DOWN
    if rand == 3:
        direction =  constants.LEFT
    if rand == 4:
        direction = constants.RIGHT
    #print("Rand = ", rand)
    #self.pacman.node = self.pacman.getNewTarget(direction)
    current_nod = self.pacman.node
    best_length = 1000000
    next_node = current_nod
    for pellet in self.pellets.pelletList:
        for nod in game_map.nodesLUT:
            if pellet.coord == game_map.nodesLUT[nod].coords:
                this_node, this_length = next_step(self.pacman.node, game_map.nodesLUT[nod], game_map)
                if this_length < best_length:
                    next_node = this_node
                    best_length = this_length
                break
    
    #self.pacman.node = next_step(self.pacman.node, self.pellets.pelletList[0], game_map)
    self.pacman.node = next_node
    #self.pacman.setPosition()
    #print("Current: ", current_nod.coords, "Next: ", self.pacman.node.coords,"Goal: " ,self.pellets.pelletList[0].coord, "Length = ", best_length)
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


def move_ghost(self, game_map, ghost_pos, pacman_pos, flee_mode):
    """Move ghost toward (or away from) Pacman."""
    # neighbors = game_map[ghost_pos].get('neighbors', [])
    # if not neighbors:
    #     return ghost_pos
    
    # Simple strategy: move toward/away from Pacman
    if flee_mode:
        # Run away from Pacman
        target = ghost_pos
        best_dist = -1

        for n in ghost_pos.node.neighbors.values():
            if n is not None:
                d = abs(n.coords[0] - pacman_pos.coords[0]) + abs(n.coords[1] - pacman_pos.coords[1])
                if d > best_dist:
                    best_dist = d
                    target = n
        num_steps = best_dist
        next_node = target

    else:
        # Chase Pacman
        #print("The ghost_pos object is\n ", ghost_pos)
        next_node, num_steps = next_step(ghost_pos.node, self.pacman.node, game_map)
    #dx = next_node.node.coords[0] - ghost_pos.node.coords[0]
    #dy = next_node.node.coords[1] - ghost_pos.node.coords[1]
    #print(ghost_pos, " current ghost node is ", ghost_pos.node.coords, "new node is ", next_node.coords, "num_steps = ", num_steps)
    ghost_pos.node = next_node


    """ if dx == 0 and dy == 0:
        direction = None
    elif dx == 0 and dy == -1:
        direction = constants.UP
    elif dx == 0 and dy == 1:
        direction = constants.DOWN
    elif dx == -1 and dy == 0:
        direction = constants.LEFT
    else:
        direction = constants.RIGHT
    """
    # i dont understand how your code updates their location to go to the
    # next ghost position, i legit have no clue bruh
    #self.ghosts.node = self.ghosts.getNewTarget(direction)
    #self.ghosts.setPosition()
    return ghost_pos.node.coords