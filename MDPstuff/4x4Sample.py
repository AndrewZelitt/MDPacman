import itertools
import numpy as np


GRID_W = 4
GRID_H = 4
BLOCKED = {(1,1), (1,2), (2,1), (2,2)}

ACTIONS = {
    "U": (-1, 0),
    "D": ( 1, 0),
    "L": ( 0,-1),
    "R": ( 0, 1),
}

REWARD_PELLET = 1.0            # reward when Pac-Man eats a pellet
REWARD_COLLISION = -20.0       # collision penalty — MAKE MORE NEGATIVE to reduce crash frequency
REWARD_STEP = -0.04            # living cost — MAKE MORE NEGATIVE to force fast pellet collection
REWARD_WIN = 10.0              # win reward after final pellet

GAMMA = 0.95                   # discount factor — CHANGE THIS to tune planning horizon
THRESH = 1e-5                  # VI convergence threshold
MAX_ITERS = 2000               # safety stop


def legal_neighbors(x,y):
    for dx,dy in ACTIONS.values():
        nx,ny = x+dx, y+dy
        if 0 <= nx < GRID_H and 0 <= ny < GRID_W and (nx,ny) not in BLOCKED:
            yield (nx,ny)

def move_deterministic(x,y,action):
    dx, dy = ACTIONS[action]
    nx, ny = x+dx, y+dy
    if 0 <= nx < GRID_H and 0 <= ny < GRID_W and (nx,ny) not in BLOCKED:
        return (nx,ny)
    return (x,y)

# pellet index mapping
PELLET_POSITIONS = [
    (i,j) for i in range(GRID_H) for j in range(GRID_W)
    if (i,j) not in BLOCKED
]
PELLET_INDEX = {pos:k for k,pos in enumerate(PELLET_POSITIONS)}
NUM_PELLETS = len(PELLET_POSITIONS)

ALL_STATES = []
for px in range(GRID_H):
    for py in range(GRID_W):
        if (px,py) in BLOCKED: continue
        for gx in range(GRID_H):
            for gy in range(GRID_W):
                if (gx,gy) in BLOCKED: continue
                for pellet_mask in range(1<<NUM_PELLETS):
                    ALL_STATES.append((px,py,gx,gy,pellet_mask))

ALL_STATES = tuple(ALL_STATES)
INDEX = {s:i for i,s in enumerate(ALL_STATES)}

def is_terminal(state):
    px,py,gx,gy,pm = state
    if (px,py)==(gx,gy):
        return True
    if pm == 0:  # no pellets remaining
        return True
    return False

def transitions(state, action):
    """Returns list of (prob, next_state, reward)."""
    px,py,gx,gy,mask = state

    if is_terminal(state):
        return [(1.0, state, 0.0)]

    npx, npy = move_deterministic(px,py,action)

    # Pellets:
    new_mask = mask
    if (npx,npy) in PELLET_INDEX:
        bit = PELLET_INDEX[(npx,npy)]
        if (mask >> bit) & 1:
            new_mask = mask & ~(1<<bit)  # remove pellet

    ghost_neighbors = list(legal_neighbors(gx,gy))
    P = 1/len(ghost_neighbors)

    outcomes = []
    for ngx,ngy in ghost_neighbors:

        # collision check
        if (npx,npy)==(ngx,ngy):
            outcomes.append(
                (P, (npx,npy,ngx,ngy,new_mask), REWARD_COLLISION)
            )
            continue

        # win check (all pellets consumed)
        if new_mask == 0:
            outcomes.append(
                (P, (npx,npy,ngx,ngy,new_mask), REWARD_WIN)
            )
            continue

        # pellet reward or step cost
        reward = REWARD_STEP
        if new_mask != mask:
            reward += REWARD_PELLET

        outcomes.append(
            (P, (npx,npy,ngx,ngy,new_mask), reward)
        )

    return outcomes

def value_iteration():
    V = np.zeros(len(ALL_STATES))
    policy = {s:"U" for s in ALL_STATES}

    for it in range(MAX_ITERS):
        delta = 0.0
        newV = np.copy(V)

        for s in ALL_STATES:

            if is_terminal(s):
                newV[INDEX[s]] = 0.0
                continue

            best_val = -1e18
            best_a = None

            for a in ACTIONS:
                val = 0.0
                for prob, ns, rew in transitions(s,a):
                    val += prob * (rew + GAMMA * V[INDEX[ns]])

                if val > best_val:
                    best_val = val
                    best_a = a

            newV[INDEX[s]] = best_val
            policy[s] = best_a
            delta = max(delta, abs(V[INDEX[s]] - best_val))

        V = newV

        if delta < THRESH:
            break

    return V, policy


V, policy = value_iteration()

print("Value iteration complete.")
print("Example policy lookup for state (Pac=2,3 Ghost=3,2 pellet_mask=all pellets):")

full_mask = (1<<NUM_PELLETS)-1
s = (2,3,3,2,full_mask)
print("Optimal action:", policy[s])
print("Value:", V[INDEX[s]])


import random

def simulate_episode(start_pac, start_ghost, policy, max_steps=200):
    # start_pac = (px, py)
    # start_ghost = (gx, gy)

    # Initial pellet mask: all pellets present
    pellet_mask = (1 << NUM_PELLETS) - 1

    px, py = start_pac
    gx, gy = start_ghost

    total_reward = 0.0
    steps = 0

    # Episode log (optional)
    # trajectory = []

    while steps < max_steps:
        state = (px, py, gx, gy, pellet_mask)

        # Check terminal right away
        if is_terminal(state):
            break

        # Pac-Man chooses optimal action from the policy
        action = policy[state]

        # Pac-Man moves
        npx, npy = move_deterministic(px, py, action)

        # Handle pellet collection
        new_mask = pellet_mask
        pellet_reward = 0.0
        if (npx, npy) in PELLET_INDEX:
            bit = PELLET_INDEX[(npx, npy)]
            if (pellet_mask >> bit) & 1:
                new_mask = pellet_mask & ~(1 << bit)
                pellet_reward = REWARD_PELLET

        # Ghost moves uniformly at random
        ghost_neighbors = list(legal_neighbors(gx, gy))
        ngx, ngy = random.choice(ghost_neighbors)

        # Collision check
        if (npx, npy) == (ngx, ngy):
            total_reward += REWARD_COLLISION
            steps += 1
            return {
                "start_pac": start_pac,
                "start_ghost": start_ghost,
                "steps": steps,
                "total_reward": total_reward,
                "ended_by": "collision",
            }

        # Win condition
        if new_mask == 0:
            total_reward += REWARD_WIN
            steps += 1
            return {
                "start_pac": start_pac,
                "start_ghost": start_ghost,
                "steps": steps,
                "total_reward": total_reward,
                "ended_by": "win",
            }

        # Add step cost and pellet reward
        total_reward += REWARD_STEP + pellet_reward

        # Update state
        px, py = npx, npy
        gx, gy = ngx, ngy
        pellet_mask = new_mask
        steps += 1

    # If it hits max_steps, treat as timeout
    return {
        "start_pac": start_pac,
        "start_ghost": start_ghost,
        "steps": steps,
        "total_reward": total_reward,
        "ended_by": "timeout",
    }
# After computing V, policy:

for ep in range(10):
    start_pac = (random.randint(0,3), random.randint(0,3))
    while start_pac in BLOCKED:
        start_pac = (random.randint(0,3), random.randint(0,3))

    start_ghost = (random.randint(0,3), random.randint(0,3))
    while start_ghost in BLOCKED:
        start_ghost = (random.randint(0,3), random.randint(0,3))

    result = simulate_episode(start_pac, start_ghost, policy)
    print(f"Episode {ep}: {result}")
