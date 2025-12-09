"""
MDP-based Pacman Policy Generator

This module combines the MDP value iteration framework with the Pacman simulator
to generate optimal moves for Pacman based on the current game state.
"""

import numpy as np
from collections import defaultdict
import constants


class PacmanMDP:
    """
    Markov Decision Process for Pacman that computes optimal policy using value iteration.
    
    State: (pacman_pos, ghost_positions, pellet_positions, power_mode_turns)
    Actions: UP, DOWN, LEFT, RIGHT
    Rewards: +10 for pellet, +50 for power pellet, +200 for eating ghost, -500 for collision with ghost
    """
    
    def __init__(self, nodes, ghost_count=4, gamma=0.99, p_success=0.9, max_iterations=100, convergence_threshold=0.01):
        """
        Initialize the Pacman MDP.
        
        Args:
            nodes: NodeGroup object containing the maze structure
            ghost_count: Number of ghosts in the game
            gamma: Discount factor for future rewards
            p_success: Probability of successful action execution
            max_iterations: Maximum iterations for value iteration
            convergence_threshold: Convergence threshold for value iteration
        """
        self.nodes = nodes
        self.ghost_count = ghost_count
        self.gamma = gamma
        self.p_success = p_success
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # All possible actions for Pacman
        self.actions = [constants.UP, constants.DOWN, constants.LEFT, constants.RIGHT]
        self.action_names = ['U', 'D', 'L', 'R']
        
        # Value function and policy cache
        self.value_function = {}
        self.policy = {}
        self.state_cache = {}
        
    def get_valid_actions(self, node):
        """Get valid actions from a given node."""
        valid_actions = []
        
        if node.neighbors.get(constants.UP) is not None:
            valid_actions.append(constants.UP)
        if node.neighbors.get(constants.DOWN) is not None:
            valid_actions.append(constants.DOWN)
        if node.neighbors.get(constants.LEFT) is not None:
            valid_actions.append(constants.LEFT)
        if node.neighbors.get(constants.RIGHT) is not None:
            valid_actions.append(constants.RIGHT)
        
        return valid_actions
    
    def calculate_reward(self, pacman_node, pellets, ghosts, power_mode, next_pacman_node, 
                         next_pellets, next_ghosts, next_power_mode):
        """
        Calculate the immediate reward for a state transition.
        
        Args:
            pacman_node: Current pacman node
            pellets: Current pellet list
            ghosts: Current ghost list
            power_mode: Current power mode turns remaining
            next_pacman_node: Next pacman node
            next_pellets: Next pellet list
            next_ghosts: Next ghost list
            next_power_mode: Next power mode turns remaining
            
        Returns:
            float: Reward value
        """
        reward = -0.1  # Small penalty for each step to encourage efficiency
        
        # Check if Pacman collects a pellet
        pellet_collected = False
        for pellet in pellets:
            if pellet.coord == pacman_node.coords:
                if pellet.name == constants.POWERPELLET:
                    reward += 50
                else:
                    reward += 10
                pellet_collected = True
                break
        
        # Check for ghost collisions
        for ghost in ghosts:
            if ghost.node.coords == pacman_node.coords:
                if power_mode > 0:
                    reward += 200  # Ate ghost
                else:
                    reward -= 500  # Hit by ghost (terminal penalty)
        
        # Bonus for clearing all pellets
        if len(next_pellets) == 0 and len(pellets) > 0:
            reward += 1000
        
        return reward
    
    def get_next_states(self, pacman_node, pellets, ghosts, power_mode):
        """
        Get all possible next states and their probabilities from current state.
        
        Args:
            pacman_node: Current pacman node
            pellets: List of pellets
            ghosts: List of ghosts
            power_mode: Current power mode turns
            
        Returns:
            dict: {action: [(prob, next_state, reward), ...]}
        """
        transitions = {}
        valid_actions = self.get_valid_actions(pacman_node)
        
        for action in valid_actions:
            transitions[action] = []
            
            # Deterministic next node for Pacman
            next_node = pacman_node.neighbors.get(action, pacman_node)
            if next_node is None:
                next_node = pacman_node
            
            # For this simplified version, we treat transitions as deterministic
            # In a full implementation, you'd add stochastic ghost movements
            prob = 1.0
            
            # Calculate reward (simplified - doesn't account for full ghost dynamics)
            reward = self.calculate_reward(pacman_node, pellets, ghosts, power_mode, 
                                          next_node, pellets, ghosts, power_mode)
            
            state_tuple = (next_node.coords, tuple(p.coord for p in pellets), 
                          tuple(g.node.coords for g in ghosts), power_mode)
            
            transitions[action].append((prob, state_tuple, reward))
        
        return transitions
    
    def discretize_state(self, pacman_node, pellets, ghosts, power_mode):
        """Convert game state to a hashable state representation."""
        pellet_tuple = tuple(sorted([p.coord for p in pellets]))
        ghost_tuple = tuple(sorted([g.node.coords for g in ghosts]))
        return (pacman_node.coords, pellet_tuple, ghost_tuple, power_mode)
    
    def get_best_action_from_node(self, pacman_node, pellets, ghosts, power_mode, 
                                   value_function=None):
        """
        Get the best action for Pacman using simple greedy/value-based approach.
        
        This uses a simplified value iteration that doesn't require full state exploration.
        Instead, it evaluates immediate and near-term rewards for each valid action.
        
        Args:
            pacman_node: Current pacman node
            pellets: Current pellet list
            ghosts: Current ghost list
            power_mode: Current power mode turns
            value_function: Optional pre-computed value function
            
        Returns:
            int: Best action (UP, DOWN, LEFT, or RIGHT)
        """
        valid_actions = self.get_valid_actions(pacman_node)
        
        if not valid_actions:
            return constants.UP  # Fallback
        
        best_action = valid_actions[0]
        best_value = -float('inf')
        
        # Evaluate each valid action
        for action in valid_actions:
            next_node = pacman_node.neighbors[action]
            if next_node is None:
                continue
            
            # Calculate action value
            action_value = self._evaluate_action(next_node, pacman_node, pellets, ghosts, power_mode)
            
            if action_value > best_value:
                best_value = action_value
                best_action = action
        
        return best_action
    
    def _evaluate_action(self, next_node, current_node, pellets, ghosts, power_mode):
        """
        Evaluate the value of an action based on immediate rewards and heuristics.
        
        Args:
            next_node: Node that Pacman would move to
            current_node: Current Pacman node
            pellets: List of pellets
            ghosts: List of ghosts
            power_mode: Current power mode turns
            
        Returns:
            float: Estimated value of this action
        """
        value = 0.0
        
        # Immediate rewards
        for pellet in pellets:
            if pellet.coord == next_node.coords:
                if pellet.name == constants.POWERPELLET:
                    value += 50
                else:
                    value += 10
        
        # Ghost collision penalty/reward
        for ghost in ghosts:
            # Some ghosts may not have a node assigned during simulation; skip them
            if getattr(ghost, 'node', None) is None:
                continue
            if ghost.node is None:
                continue
            if ghost.node.coords == next_node.coords:
                if power_mode > 0:
                    value += 200
                else:
                    value -= 500
        
        # Proximity-based rewards
        # Reward pellet proximity (unless power mode is active)
        if power_mode == 0 and pellets:
            min_distance = float('inf')
            for pellet in pellets:
                dist = self._manhattan_distance(next_node.coords, pellet.coord)
                min_distance = min(min_distance, dist)
            
            # Reward moving toward pellets
            current_min_dist = float('inf')
            for pellet in pellets:
                dist = self._manhattan_distance(current_node.coords, pellet.coord)
                current_min_dist = min(current_min_dist, dist)
            
            if min_distance < current_min_dist:
                value += 5.0 / (1.0 + min_distance)
        
        # In power mode, approach ghosts
        if power_mode > 0 and ghosts:
            min_distance = float('inf')
            for ghost in ghosts:
                if getattr(ghost, 'node', None) is None:
                    continue
                if ghost.node is None:
                    continue
                dist = self._manhattan_distance(next_node.coords, ghost.node.coords)
                min_distance = min(min_distance, dist)
            
            # Reward moving toward ghosts
            current_min_dist = float('inf')
            for ghost in ghosts:
                if getattr(ghost, 'node', None) is None:
                    continue
                if ghost.node is None:
                    continue
                dist = self._manhattan_distance(current_node.coords, ghost.node.coords)
                current_min_dist = min(current_min_dist, dist)
            
            if min_distance < current_min_dist:
                value += 10.0 / (1.0 + min_distance)
        
        # Avoid collision with non-power-mode ghosts
        if power_mode == 0 and ghosts:
            min_ghost_distance = float('inf')
            for ghost in ghosts:
                if getattr(ghost, 'node', None) is None:
                    continue
                if ghost.node is None:
                    continue
                dist = self._manhattan_distance(next_node.coords, ghost.node.coords)
                min_ghost_distance = min(min_ghost_distance, dist)
            
            # Penalty for being close to ghosts
            if min_ghost_distance <= 2:
                value -= 50.0 / (1.0 + min_ghost_distance)
        
        return value
    
    @staticmethod
    def _manhattan_distance(coord1, coord2):
        """Calculate Manhattan distance between two coordinates."""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])
    
    def compute_policy_for_state(self, pacman_node, pellets, ghosts, power_mode):
        """
        Compute the optimal policy (best action) for the current game state.
        
        Args:
            pacman_node: Current Pacman node
            pellets: List of pellets remaining
            ghosts: List of ghosts
            power_mode: Turns remaining in power mode
            
        Returns:
            int: Best action for Pacman to take
        """
        return self.get_best_action_from_node(pacman_node, pellets, ghosts, power_mode)


class SimplePacmanMDPSolver:
    """
    Simplified MDP solver for Pacman that doesn't require full state enumeration.
    Uses greedy evaluation based on current state only.
    """
    
    def __init__(self, nodes, gamma=0.99):
        """
        Initialize the simplified solver.
        
        Args:
            nodes: NodeGroup object
            gamma: Discount factor
        """
        self.mdp = PacmanMDP(nodes, gamma=gamma)
    
    def get_next_move(self, pacman_node, pellets, ghosts, power_mode):
        """
        Get the next optimal move for Pacman.
        
        Args:
            pacman_node: Current Pacman node
            pellets: List of pellets
            ghosts: List of ghosts
            power_mode: Turns remaining in power mode
            
        Returns:
            int: Action constant (UP, DOWN, LEFT, or RIGHT)
        """
        return self.mdp.compute_policy_for_state(pacman_node, pellets, ghosts, power_mode)
