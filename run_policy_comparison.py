"""
Script to run simulations and compare MDP policy vs baseline policies.

Usage:
    1. Run the game: python run.py
    2. Press 'S' to run simulations using MDP policy
    3. Run this script to see detailed analysis
    
Or run standalone:
    python run_policy_comparison.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import pygame
from mdp_value_iteration import ValueIterationPacmanMDP
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from pacman import Pacman
from simulator import run_pacman_simulation, move_pacman, move_ghost, next_step
from constants import *
import random
import time


class SimulationComparator:
    """Compare different Pacman policies."""
    
    def __init__(self, num_runs=30, max_moves=10000):
        self.num_runs = num_runs
        self.max_moves = max_moves
        self.nodes = NodeGroup("maze1.txt")
        
           # Initialize pygame first (required for sprites)
        pygame.init()
        pygame.display.set_mode((100, 100))
        
        # Initialize value-iteration solver instance (kept for convenience)
        from mdp_value_iteration import ValueIterationPacmanMDP
        self.mdp_solver = ValueIterationPacmanMDP(self.nodes)
        
    def create_game_state(self):
        """Create a fresh game state for simulation."""
        nodes = NodeGroup("maze1.txt")
        nodes.setPortalPair((0, 17), (27, 17))
        
        pacman = Pacman(nodes.getNodeFromTiles(15, 26))
        pellets = PelletGroup("maze1.txt", nodes)
        ghosts = GhostGroup(nodes.getStartTempNode(), pacman)
        ghosts.randomizeSpawn(nodes, exclude_coords={(15.0, 26.0)})
        
        return pacman, pellets, ghosts, nodes
    
    def run_mdp_simulation(self):
        """Run simulation with MDP policy."""
        scores = []
        
        print("\n" + "="*60)
        print("MDP POLICY SIMULATION")
        print("="*60)
        
        for run_num in range(self.num_runs):
            pacman, pellets, ghosts, nodes = self.create_game_state()
            
            score = self._simulate_mdp_moves(pacman, pellets, ghosts, nodes)
            scores.append(score)
            
            print(f"Run {run_num + 1}: Score = {score} | Pellets left: {len(pellets.pelletList)-4}")
        
        return scores
    
    def run_random_simulation(self):
        """Run simulation with random moves (baseline)."""
        scores = []
        
        print("\n" + "="*60)
        print("RANDOM POLICY SIMULATION (Baseline)")
        print("="*60)
        
        for run_num in range(self.num_runs):
            pacman, pellets, ghosts, nodes = self.create_game_state()
            
            score = self._simulate_random_moves(pacman, pellets, ghosts, nodes)
            scores.append(score)
            
            print(f"Run {run_num + 1}: Score = {score} | Pellets left: {len(pellets.pelletList)-4}")
        
        return scores
    
    def run_greedy_nearest_pellet_simulation(self):
        """Run simulation with greedy nearest-pellet policy."""
        scores = []
        
        print("\n" + "="*60)
        print("GREEDY NEAREST-PELLET POLICY (Baseline)")
        print("="*60)
        
        for run_num in range(self.num_runs):
            pacman, pellets, ghosts, nodes = self.create_game_state()
            
            score = self._simulate_greedy_pellet_moves(pacman, pellets, ghosts, nodes)
            scores.append(score)
            
            print(f"Run {run_num + 1}: Score = {score} | Pellets left: {len(pellets.pelletList)-4}")
        
        return scores
    
    def _simulate_mdp_moves(self, pacman, pellets, ghosts, nodes):
        """Simulate with MDP policy."""
        score = 0
        power_mode = 0
        moves = 0
        
        # Use value-iteration based solver (recomputed each step)
        mdp_solver = ValueIterationPacmanMDP(nodes)
        
        while moves < self.max_moves and len(pellets.pelletList) > 0:
            # Collect pellets
            pellets_to_remove = []
            
            for pellet in pellets.pelletList:
                if pacman.node.coords == pellet.coord:
                #if pacman.collideCheck(pellet):
                    if pellet.name == POWERPELLET:
                        power_mode = 20
                        score += 50
                    else:
                        score += 10
                    pellets_to_remove.append(pellet)
            """pellet = pacman.eatPellets(pellets.pelletList)
            #if pellet is not None:
                if pellet.name == POWERPELLET:
                    power_mode = 20
                    score += 50
                else:
                    score += 10
                pellets_to_remove.append(pellet)"""
            for pellet in pellets_to_remove:
                pellets.pelletList.remove(pellet)
            
            # Check ghost collisions
            for ghost in ghosts:
                if getattr(ghost, 'node', None) is None:
                    continue
                if pacman.node.coords == ghost.node.coords:
                    if power_mode > 0:
                        score += 200
                        power_mode -= 1
                    else:
                        return score
            
            # Get MDP move using value iteration (recomputes policy against current ghost positions)
            best_action = mdp_solver.compute_best_action(pacman.node, pellets.pelletList, ghosts)
            next_node = pacman.node.neighbors.get(best_action)
            if next_node is not None:
                pacman.node = next_node
                #pacman.setPosition()
            
            # Move ghosts
            for ghost in ghosts:
                if getattr(ghost, 'node', None) is None:
                    continue
                next_ghost_node, _ = next_step(ghost.node, pacman.node, nodes)
                ghost.node = next_ghost_node
            
            if power_mode > 0:
                power_mode -= 1
            
            moves += 1
            
            if len(pellets.pelletList) == 0:
                score += 1000
                break
        
        return score
    
    def _simulate_random_moves(self, pacman, pellets, ghosts, nodes):
        """Simulate with random moves."""
        score = 0
        power_mode = 0
        moves = 0
        
        while moves < self.max_moves and len(pellets.pelletList) > 0:
            # Collect pellets
            pellets_to_remove = []
            for pellet in pellets.pelletList:
                if pacman.node.coords == pellet.coord:
                    if pellet.name == POWERPELLET:
                        power_mode = 20
                        score += 50
                    else:
                        score += 10
                    pellets_to_remove.append(pellet)
            
            for pellet in pellets_to_remove:
                pellets.pelletList.remove(pellet)
            
            # Check ghost collisions
            for ghost in ghosts:
                if getattr(ghost, 'node', None) is None:
                    continue
                if pacman.node.coords == ghost.node.coords:
                    if power_mode > 0:
                        score += 200
                        power_mode -= 1
                    else:
                        return score
            
            # Random move
            valid_actions = []
            if pacman.node.neighbors.get(UP) is not None:
                valid_actions.append(UP)
            if pacman.node.neighbors.get(DOWN) is not None:
                valid_actions.append(DOWN)
            if pacman.node.neighbors.get(LEFT) is not None:
                valid_actions.append(LEFT)
            if pacman.node.neighbors.get(RIGHT) is not None:
                valid_actions.append(RIGHT)
            
            if valid_actions:
                action = random.choice(valid_actions)
                next_node = pacman.node.neighbors.get(action)
                if next_node is not None:
                    pacman.node = next_node
            
            # Move ghosts
            for ghost in ghosts:
                if getattr(ghost, 'node', None) is None:
                    continue
                next_ghost_node, _ = next_step(ghost.node, pacman.node, nodes)
                ghost.node = next_ghost_node
            
            if power_mode > 0:
                power_mode -= 1
            
            moves += 1
            
            if len(pellets.pelletList) == 0:
                score += 1000
                break
        
        return score
    
    def _simulate_greedy_pellet_moves(self, pacman, pellets, ghosts, nodes):
        """Simulate with greedy nearest-pellet strategy."""
        score = 0
        power_mode = 0
        moves = 0
        
        while moves < self.max_moves and len(pellets.pelletList) > 0:
            # Collect pellets
            pellets_to_remove = []
            for pellet in pellets.pelletList:
                if pacman.node.coords == pellet.coord:
                    if pellet.name == POWERPELLET:
                        power_mode = 20
                        score += 50
                    else:
                        score += 10
                    pellets_to_remove.append(pellet)
            
            for pellet in pellets_to_remove:
                pellets.pelletList.remove(pellet)
            
            # Check ghost collisions
            for ghost in ghosts:
                if getattr(ghost, 'node', None) is None:
                    continue
                if pacman.node.coords == ghost.node.coords:
                    if power_mode > 0:
                        score += 200
                        power_mode -= 1
                    else:
                        return score
            
            # Greedy: move toward nearest pellet
            best_length = 1000000
            next_node = pacman.node
            
            for pellet in pellets.pelletList:
                for nod_key in nodes.nodesLUT:
                    if pellet.coord == nodes.nodesLUT[nod_key].coords:
                        goal_node = nodes.nodesLUT[nod_key]
                        this_node, this_length = next_step(pacman.node, goal_node, nodes)
                        if this_length < best_length:
                            next_node = this_node
                            best_length = this_length
                        break
            
            pacman.node = next_node
            
            # Move ghosts
            for ghost in ghosts:
                if getattr(ghost, 'node', None) is None:
                    continue
                next_ghost_node, _ = next_step(ghost.node, pacman.node, nodes)
                ghost.node = next_ghost_node
            
            if power_mode > 0:
                power_mode -= 1
            
            moves += 1
            
            if len(pellets.pelletList) == 0:
                score += 1000
                break
        
        return score
    
    def compare_and_print_results(self):
        """Run all simulations and print comparison."""
        mdp_scores = self.run_mdp_simulation()
        random_scores = self.run_random_simulation()
        greedy_scores = self.run_greedy_nearest_pellet_simulation()
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        def print_stats(name, scores):
            avg = sum(scores) / len(scores)
            best = max(scores)
            worst = min(scores)
            print(f"\n{name}:")
            print(f"  Average Score: {avg:.1f}")
            print(f"  Best Score:    {best}")
            print(f"  Worst Score:   {worst}")
            print(f"  Scores:        {scores}")
        
        print_stats("MDP Policy", mdp_scores)
        print_stats("Random Policy", random_scores)
        print_stats("Greedy Nearest-Pellet Policy", greedy_scores)
        
        # Calculate improvements
        mdp_avg = sum(mdp_scores) / len(mdp_scores)
        random_avg = sum(random_scores) / len(random_scores)
        greedy_avg = sum(greedy_scores) / len(greedy_scores)
        
        print("\n" + "="*60)
        print("IMPROVEMENTS")
        print("="*60)
        print(f"MDP vs Random:       {(mdp_avg - random_avg) / max(random_avg, 1):.1%} improvement")
        print(f"MDP vs Greedy:       {(mdp_avg - greedy_avg) / max(greedy_avg, 1):.1%} improvement")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Pacman policies")
    parser.add_argument("--runs", type=int, default=75, help="Number of simulation runs per policy")
    parser.add_argument("--max-moves", type=int, default=10000, help="Maximum moves per simulation")
    
    args = parser.parse_args()
    
    comparator = SimulationComparator(num_runs=args.runs, max_moves=args.max_moves)
    
    start_time = time.time()
    comparator.compare_and_print_results()
    end_time = time.time()
    
    print(f"\n\nTotal execution time: {end_time - start_time:.2f} seconds")
