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
                    self.pellets.pelletList.pop(pellet)
                    power_mode = 1
                else:
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
        
        
       

        # Check ghost collisions
        if pacman_pos in ghost_positions:
            if power_mode > 0:
                score += 200  # Eat ghost
                ghost_idx = ghost_positions.index(pacman_pos)
                ghost_positions[ghost_idx].node = self.nodes.getNodeFromTiles(2+11.5, 3+14)
            else:
               return score
        """     
        
        # Move Pacman (simple strategy: move toward nearest pellet)
        # this will be replaced with the mdp stuff
        pacman_pos = move_pacman(self, game_map, pacman_pos, pellets_remaining)

        # Move ghosts (simple chase strategy)
        [move_ghost(self, game_map, ghost_pos, pacman_pos, power_mode > 0) for ghost_pos in ghost_positions]

        # Decrease power mode
        if power_mode > 0:
            power_mode -= 1
        
        moves += 1

        """self.clock.tick(30) / 1000.0
        self.screen.blit(self.background, (0, 0))
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)
        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = constants.SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))
        events = pygame.event.get()
        pygame_widgets.update(events)
        pygame.display.update()"""

    
    # Bonus for completing level
    if pellets_remaining == 0:
        score += 1000
    
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
    self.pacman.node = next_step(self.pacman.node, self.pellets.pelletList[0], game_map)
    #self.pacman.setPosition()
    print("Current: ", current_nod.coords, "Next: ", self.pacman.node.coords,"Goal: " ,self.pellets.pelletList[0].coord)
    return self.pacman.node


def next_step(start, goal, game_map):
    """this is a helper function that finds the next position for the ghosts to be in"""
    if start == goal:
        return start

    queue = deque([start])
    visited = {start: None}

    while queue:
        current = queue.popleft()
        for node in game_map.nodesLUT.keys():
            for neighbor in game_map.nodesLUT[node].neighbors.values():
                if neighbor is not None:
                    #print(node, "has neighbor: ", neighbor.coords)
                    if neighbor not in visited:
                        visited[neighbor] = current
                        queue.append(neighbor)

                        if neighbor == goal:
                            path = [neighbor]
                            while visited[path[-1]] is not None:
                                path.append(visited[path[-1]])
                            path.reverse()

                            if len(path) > 1:
                                return path[1]
                            return start

    return start  # failsafe but like this sohuld never happen


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

        for n in game_map[ghost_pos].get('neighbors', []):
            d = abs(n[0] - pacman_pos[0]) + abs(n[1] - pacman_pos[1])
            if d > best_dist:
                best_dist = d
                target = n
        next_node = target

    else:
        # Chase Pacman
        #print("The ghost_pos object is\n ", ghost_pos)
        next_node = next_step(ghost_pos.node, self.pacman.node, game_map)
    
    #dx = next_node.node.coords[0] - ghost_pos.node.coords[0]
    #dy = next_node.node.coords[1] - ghost_pos.node.coords[1]
    #print(ghost_pos, " current ghost node is ", ghost_pos.node.coords, "new node is ", next_node.coords)
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