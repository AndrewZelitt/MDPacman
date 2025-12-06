import nodes
import constants
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
        print(pacman_pos)

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
                ghost_positions[ghost_idx] = ghost_starts[ghost_idx]  # Respawn ghost
            else:
               return score
        """        
        
        # Move Pacman (simple strategy: move toward nearest pellet)
        # this will be replaced with the mdp stuff
        pacman_pos = move_pacman(self, game_map, pacman_pos, pellets_remaining)
        

        """
        # Move ghosts (simple chase strategy)
        ghost_positions = [move_ghost(game_map, ghost_pos, pacman_pos, power_mode > 0) 
                          for ghost_pos in ghost_positions]
        """
        # Decrease power mode
        if power_mode > 0:
            power_mode -= 1
        
        moves += 1
    
    # Bonus for completing level
    if pellets_remaining == 0:
        score += 1000
    
    return score


def move_pacman(self, game_map, current_pos, pellets_remaining):
    import random
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
    print("Rand = ", rand)
    self.pacman.node = self.pacman.getNewTarget(direction)
    self.pacman.setPosition()
    return self.pacman.node.coords

    


def move_ghost(self, game_map, ghost_pos, pacman_pos, flee_mode):
    """Move ghost toward (or away from) Pacman."""
    neighbors = game_map[ghost_pos].get('neighbors', [])
    if not neighbors:
        return ghost_pos
    
    # Simple strategy: move toward/away from Pacman
    if flee_mode:
        # Run away from Pacman
        return max(neighbors, key=lambda pos: abs(pos[0] - pacman_pos[0]) + abs(pos[1] - pacman_pos[1]))
    else:
        # Chase Pacman
        return min(neighbors, key=lambda pos: abs(pos[0] - pacman_pos[0]) + abs(pos[1] - pacman_pos[1]))