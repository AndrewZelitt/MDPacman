"""
Value-iteration based MDP solver for Pacman (node-based abstraction).

Approach:
- State: Pacman node (each node in NodeGroup)
- Actions: move to neighboring node (UP/DOWN/LEFT/RIGHT/PORTAL)
- Reward(s->s') = pellet reward at s' - danger_penalty(s') - step_penalty
- Danger penalty computed from shortest-node distance from any ghost to that node

The solver recomputes rewards/danger map each decision step (so it adapts to moving ghosts),
then runs standard value iteration over the node graph (small: number of nodes in maze),
and returns the best action for the current pacman node.

This is an approximation (ghosts are summarized by a static danger map for the current step),
but it's a correct Bellman-style value iteration over the abstracted state space and
should produce safer policies than a one-step heuristic.
"""
from collections import deque
import constants


class ValueIterationPacmanMDP:
    def __init__(self, nodegroup, gamma=0.95, step_penalty=-0.01, danger_weight=15.0, collision_penalty=-200.0,
                 pellet_reward=150.0, power_pellet_reward=2000.0, ghost_bonus=400.0,
                 vulnerable_ghost_bonus=5000.0, max_iters=400, tol=1e-4):
        self.nodegroup = nodegroup
        self.gamma = gamma
        self.step_penalty = step_penalty
        self.danger_weight = danger_weight
        self.collision_penalty = collision_penalty
        self.pellet_reward = pellet_reward
        self.power_pellet_reward = power_pellet_reward
        self.ghost_bonus = ghost_bonus
        # Bonus for moving toward vulnerable ghosts
        self.vulnerable_ghost_bonus = vulnerable_ghost_bonus
        self.max_iters = max_iters
        self.tol = tol

        # Precompute list of nodes
        self.nodes = list(self.nodegroup.nodesLUT.values())
        # Map node to index
        self.node_index = {node: i for i, node in enumerate(self.nodes)}

    def _compute_ghost_distance_map(self, ghosts):
        # For each node compute min BFS distance to any ghost (in node steps)
        inf = 10 ** 9
        dist_map = {node: inf for node in self.nodes}

        for ghost in ghosts:
            if getattr(ghost, 'node', None) is None:
                continue
            start = ghost.node
            # BFS from this ghost
            queue = deque([(start, 0)])
            visited = {start}
            while queue:
                node, d = queue.popleft()
                if dist_map[node] > d:
                    dist_map[node] = d
                for neigh in node.neighbors.values():
                    if neigh is not None and neigh not in visited:
                        visited.add(neigh)
                        queue.append((neigh, d + 1))
        return dist_map

    def _pellet_reward_at(self, node, pellets, dist_map=None, ghosts=None):
        """Return pellet reward at `node`. If this is a power pellet and a `dist_map`
        and `ghosts` are provided, add a small heuristic bonus proportional to
        the number of ghosts and their proximity to the node so VI recognizes
        the strategic value of power pellets.
        """
        for pellet in pellets:
            if pellet.coord == node.coords:
                if pellet.name == constants.POWERPELLET:
                    bonus = 0.0
                    if dist_map is not None and ghosts is not None:
                        # scale bonus by number of ghosts and inverse distance
                        num_ghosts = sum(1 for _ in ghosts)
                        if num_ghosts > 0:
                            d = dist_map.get(node, 100000)
                            bonus = self.ghost_bonus * (num_ghosts / (1.0 + float(d)))
                    return self.power_pellet_reward + bonus
                return self.pellet_reward
        return 0.0

    def compute_best_action(self, pacman_node, pellets, ghosts):
        """
        Compute value iteration given current pellets and ghost positions and
        return the best action for the provided `pacman_node`.
        """
        if pacman_node is None:
            return constants.UP

        # Run value iteration and return best action
        V, dist_map, actions_map = self._run_value_iteration(pellets, ghosts)

        # Extract best action for pacman_node
        best_action = None
        best_q = -float('inf')
        for (action, neigh) in actions_map.get(pacman_node, []):
            pellet_r = self._pellet_reward_at(neigh, pellets, dist_map=dist_map, ghosts=ghosts)
            dist = dist_map.get(neigh, 100000)
            if dist == 0:
                danger = self.collision_penalty
            else:
                danger = - self.danger_weight / (1.0 + dist)
            # Bonus for moving toward vulnerable ghosts (closer is better)
            vuln_bonus = 0.0
            if ghosts is not None:
                for ghost in ghosts:
                    if getattr(ghost, 'node', None) is None:
                        continue
                    try:
                        mode = ghost.mode.current
                    except Exception:
                        mode = None
                    if mode == constants.FREIGHT:
                        dghost = dist_map.get(neigh, 100000)
                        vuln_bonus += self.vulnerable_ghost_bonus * (1.0 / (1.0 + float(dghost)))
            reward = pellet_r + danger + self.step_penalty + vuln_bonus
            j = self.node_index[neigh]
            q = reward + self.gamma * V[j]
            if q > best_q:
                best_q = q
                best_action = action

        # Fallback
        if best_action is None:
            return constants.UP
        return best_action

    def _run_value_iteration(self, pellets, ghosts):
        """Run VI and return value vector, distance map and actions map."""
        dist_map = self._compute_ghost_distance_map(ghosts)

        n = len(self.nodes)
        V = [0.0] * n

        # Precompute neighbors/actions mapping for each node
        actions_map = {}
        for node in self.nodes:
            acts = []
            for dir_key, neigh in node.neighbors.items():
                if neigh is not None:
                    acts.append((dir_key, neigh))
            actions_map[node] = acts

        # Value iteration
        for it in range(self.max_iters):
            delta = 0.0
            newV = V.copy()
            for i, node in enumerate(self.nodes):
                best_q = -float('inf')
                acts = actions_map.get(node, [])
                if not acts:
                    reward = 0.0
                    q = reward + self.gamma * V[i]
                    best_q = q
                else:
                    for (action, neigh) in acts:
                        pellet_r = self._pellet_reward_at(neigh, pellets, dist_map=dist_map, ghosts=ghosts)
                        dist = dist_map.get(neigh, 100000)
                        if dist == 0:
                            danger = self.collision_penalty
                        else:
                            danger = - self.danger_weight / (1.0 + dist)
                        # Bonus for moving toward vulnerable ghosts
                        vuln_bonus = 0.0
                        if ghosts is not None:
                            for ghost in ghosts:
                                if getattr(ghost, 'node', None) is None:
                                    continue
                                try:
                                    mode = ghost.mode.current
                                except Exception:
                                    mode = None
                                if mode == constants.FREIGHT:
                                    dghost = dist_map.get(neigh, 100000)
                                    vuln_bonus += self.vulnerable_ghost_bonus * (1.0 / (1.0 + float(dghost)))

                        reward = pellet_r + danger + self.step_penalty + vuln_bonus
                        j = self.node_index[neigh]
                        q = reward + self.gamma * V[j]
                        if q > best_q:
                            best_q = q
                newV[i] = best_q
                delta = max(delta, abs(newV[i] - V[i]))
            V = newV
            if delta < self.tol:
                break

        return V, dist_map, actions_map

    def compute_best_action_with_qs(self, pacman_node, pellets, ghosts):
        """Return (best_action, list_of_(action, q_value)) for pacman_node."""
        V, dist_map, actions_map = self._run_value_iteration(pellets, ghosts)

        q_list = []
        best_action = None
        best_q = -float('inf')
        for (action, neigh) in actions_map.get(pacman_node, []):
            pellet_r = self._pellet_reward_at(neigh, pellets, dist_map=dist_map, ghosts=ghosts)
            dist = dist_map.get(neigh, 100000)
            if dist == 0:
                danger = self.collision_penalty
            else:
                danger = - self.danger_weight / (1.0 + dist)
            reward = pellet_r + danger + self.step_penalty
            j = self.node_index[neigh]
            q = reward + self.gamma * V[j]
            q_list.append((action, q, reward, neigh))
            if q > best_q:
                best_q = q
                best_action = action

        return best_action, q_list
