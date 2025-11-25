import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import random
from automata.fa.dfa import DFA

# -------------------------------
# 1. Define the MDP environment
# -------------------------------
class GridWorldMDP:
    def __init__(self, N=10, p_success=0.8, gamma=0.95, pits=[(1,3)], goals=[(9,9)]):
        self.N = N
        self.p_success = p_success
        self.gamma = gamma
        self.actions = ['U', 'D', 'L', 'R']
        self.goal = goals
        self.pit = pits
        self.rewards = self._build_reward_map()
        self.transition_probs = self.get_transition_probs()
        self.policy= self.find_policy()
        self.states = self.get_states()
        
    def get_states(self):
        states = []
        for x in range(0, self.N):
            for y in range(0,self.N):
                states.append((x, y))
        return states
        
    def get_transition_probs(self):
         # need to make transition probabilties
        states = []
        for x in range(0, self.N):
            for y in range(0,self.N):
                states.append((x, y))

        transition_probabilities = {}
        i = 0
        for state in states:
            for action in self.actions:
                outcomes = self.transitions(state, action);
                for outcome in outcomes:
                    # insert a new transition probability.
                    n_s = outcome[1]
                    prob = outcome[0]
                    #if(state == (0,0)):
                        #print("0,0= ",outcome[1], action, outcome[0])
                    # the transition probabilites also include the reward value as this makes it possible to get the policy
                    if((state,action, n_s) in transition_probabilities):
                        new = (transition_probabilities[(state, action, n_s)][0] + prob, transition_probabilities[(state, action, n_s)][1] + outcome[2])
                        #print(transition_probabilities[(state, action, n_s)], new)

                        transition_probabilities[(state, action, n_s)] = new
                    else:
                        transition_probabilities[(state, action, n_s)] =  (prob, outcome[2])
                    i += 1
        #print(i)
        return transition_probabilities

    def _build_reward_map(self):
        R = -0.04 * np.ones((self.N, self.N))
        for goal in self.goal:
            R[goal] = +1.0
        for pit in self.pit:
            R[pit] = -1.0
        return R
    def in_bounds(self, state):
        i, j = state
        return 0 <= i < self.N and 0 <= j < self.N
    def next_state(self, state, action):
        """Return deterministic next state for given action"""
        i, j = state
        if action == 'U': i -= 1
        elif action == 'D': i += 1
        elif action == 'L': j -= 1
        elif action == 'R': j += 1
        next_s = (i, j)
        return next_s if self.in_bounds(next_s) else state
    def transitions(self, state, action):
        """
        Return list of (prob, next_state, reward) tuples
        capturing stochastic movement.
        """
        outcomes = []
        for a_alt in self.actions:
            prob = self.p_success if a_alt == action else (1 - self.p_success) / 3
            s_next = self.next_state(state, a_alt)
            r = self.rewards[s_next]
            outcomes.append((prob, s_next, r))
        return outcomes
    def find_policy(self):
        #num_states = N^2
        num_states = self.N**2
        num_actions = len(self.actions)

        c = np.ones(num_states)

        A_ub = []
        b_ub = []

        states = []
        for x in range(0, self.N):
            for y in range(0,self.N):
                states.append((x, y))

        
        for s_idx, state in enumerate(states):
            for a_idx, action in enumerate(self.actions):
                row = np.zeros(num_states)
                row[s_idx] = -1

            #print("I am in the idx loop")

            # Coefficients for the next states' values (V(s'))
            immed = 0.0
            for ns_idx, next_state in enumerate(states):
                if (state, action, next_state) in self.transition_probs:
                    prob = self.transition_probs[(state,action,next_state)][0]
                    row[ns_idx] += self.gamma * prob
                    immed += self.transition_probs[(state,action,next_state)][1]
                    
            A_ub.append(row)
            b_ub.append(-immed)


            # The right-hand side of the inequality is the immediate reward -R(s, a)
            # since the inequality is A_ub @ x <= b_ub
            # We use the state-action reward calculated previously
            
        
        A_ub = np.array(A_ub)
        # b_ub = np.array(self.rewards)
        #b_ub = np.array(self.rewards).flatten()
        b_ub = np.array(b_ub)
        bounds = [(None, None)] * num_states

        #print("Objective function coefficients (c):", c)
        #print("Inequality constraints matrix (A_ub):", A_ub)
        #print("Inequality constraints vector (b_ub):", b_ub)
        #print("Bounds for state values:", bounds)   

        # Solve the linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        # Check if the solver was successful
        if result.success:
            print("Linear programming solved successfully.")
            # The optimal value function is stored in result.x
            optimal_value_function_lp = {states[i]: result.x[i] for i in range(num_states)}
            #print("Optimal Value Function (Linear Programming):", optimal_value_function_lp)
        else:
            print("Linear programming did not converge.")
            print("Result:", result)


        optimal_policy_lp = {}

        for state in states:
            best_action = None
            best_q_value = -float('inf')

            for action in self.actions:
                q_value = 0
                for next_state in states:
                    if (state, action, next_state) in self.transition_probs:
                        prob = self.transition_probs[(state, action, next_state)][0]
                        reward = self.transition_probs[(state, action, next_state)][1]
                        q_value += prob * (reward + self.gamma * optimal_value_function_lp[next_state])
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action

            optimal_policy_lp[state] = best_action

        #print("Optimal Policy (Linear Programming):")
        #for sta in optimal_policy_lp:
            #print(sta, optimal_policy_lp[sta])
        

        return optimal_policy_lp
    def get_path(self, length, initial_state, policy):
        path = [(initial_state)]
        if not policy:
            policy = self.policy
        curr_state = initial_state
        total_reward = 0.0
        for i in range(length):
            # probabilistic next state
            act = policy[curr_state]
            i,j = curr_state
            random_float = random.random();
            if act == 'U':
                if random_float <= self.p_success :
                    i -= 1
                # biggest num
                elif random_float > self.p_success and random_float > self.p_success + 2.0*(1.0-self.p_success)/3.0 :
                    i += 1
                elif random_float > self.p_success and random_float > self.p_success + (1.0-self.p_success)/3.0 :
                    j -= 1
                elif random_float > self.p_success :
                    j += 1
            elif act == 'D':
                if random_float <= self.p_success:
                    i += 1
                elif random_float > self.p_success and random_float > self.p_success + 2.0*(1-self.p_success)/3.0 :
                    i -= 1
                elif random_float > self.p_success and random_float > self.p_success + (1.0-self.p_success)/3.0 :
                    j -= 1
                elif random_float > self.p_success :
                    j += 1
            elif act == 'L':
                if random_float <= self.p_success:
                    j -= 1
                elif random_float > self.p_success and random_float > self.p_success + 2.0*(1.0-self.p_success)/3.0 :
                    i += 1
                elif random_float > self.p_success and random_float > self.p_success + (1.0-self.p_success)/3.0 :
                    i -= 1
                elif random_float > self.p_success :
                    j += 1
            elif act == 'R':
                if random_float <= self.p_success:
                    j += 1
                elif random_float > self.p_success and random_float > self.p_success + 2.0*(1.0-self.p_success)/3.0 :
                    i -= 1
                elif random_float > self.p_success and random_float > self.p_success + (1.0-self.p_success)/3.0 :
                    j -= 1
                elif random_float > self.p_success :
                    i += 1

            #gives a random chance for each action to givea different action.
            if self.in_bounds((i,j)):
                state = (i,j)
            else:
                state = curr_state
            total_reward += self.rewards[state[0]][state[1]]
            path.append((state, self.rewards[state[0]][state[1]]))
            curr_state = state
        
        return path, total_reward
    def get_attr(self, F):
        # do things

        states = []
        for x in range(0, self.N):
            for y in range(0,self.N):
                states.append((x, y))
        Xsets = [set(F)]
        #print(Xsets)
        policy = dict([])
        #print(F)
        for s in F:
            for a in self.actions:
                for outcome in self.transitions(s,a):
                    if not (outcome[1] in self.pit):
                        policy[s] = a
                        break
        #print("Policy = ", policy)
        while True:
            X_new = set()
            for s in (set(states) - Xsets[-1]):
                for a in self.actions:
                    for val in Xsets[-1]:
                            if self.next_state(s,a) == val and self.transition_probs[(s,a,self.next_state(s,a))][0] > 0.0 :
                                X_new.add(s)
                                #print("added new X", X_new)
                                policy[s] = a
                                break
        
            new_X = Xsets[-1].union(X_new)
            if new_X == Xsets[-1]:
                #print(policy)
                return new_X, Xsets, policy

            Xsets.append(new_X)
    def attr2(self, F):
        states = []
        for x in range(0, self.N):
            for y in range(0,self.N):
                states.append((x, y))

        Y = list(set(states) - set(self.pit))

        Y_new = []

        Xsets = set(F)
        X_new = set()
        i = 0

        while(Y != Y_new):
            while(True):
                X_new = Xsets.union(self.preattr(Xsets, Y))
                if (X_new == Xsets):
                    Xsets = X_new
                    break
                Xsets = X_new
                i += 1

            Y_new = Xsets
            if(Y_new == Y):
                break
            Y = Y_new

        return Y
    def preattr(self, X, Y):
        states = []
        for x in range(0, self.N):
            for y in range(0,self.N):
                states.append((x, y))

        Apre = []

        #print("I am in the preattractor step")
        i = 0
        for state in states:
            for action in self.actions:
                for val in X:
                    if not (state in self.pit):
                        if((state,action,val) in self.transition_probs):
                            if self.transition_probs[(state,action,val)][0] > 0.0:
                                prob = 0.0
                                for sta in Y:
                                    if((state,action,sta) in self.transition_probs):
                                        prob += self.transition_probs[(state,action,sta)][0] 
                            
                                if prob >= 1.0:
                                    i += 1
                                
                                    Apre.append(state)
        #print("Elements added into APre = ",i)
        return Apre

                        


gridworld = GridWorldMDP()

print("Policy = ",gridworld.policy)

paths = []
rand_init = (int(10*random.random()), int(10*random.random()))
for i in range(10):
    path, reward = gridworld.get_path(20, rand_init,gridworld.policy)
    paths.append(path)
    print(i+1,":", paths[i])
    print(i+1, " reward =", reward, "\n")


#print("")
attr = gridworld.attr2(gridworld.goal)
#print(attr)

#print("\n",len(attr))
#print(len(gridworld.transition_probs))


#print(gridworld.next_state((0,0), 'U'))
states = []
for x in range(0, 10):
    for y in range(0,10):
        states.append((x, y))

#for i in range(10):
    #print(' '.join(gridworld.policy[(i,j)] for j in range(10)))



class TranSys:
    def __init__(self, states, actions, initial_state, trans=None):
        self.states = states
        self.actions = actions
        self.initial_state = initial_state
        if trans == None:
            self.trans = dict([])
            for s in states:
                self.trans[s] = {action: None for action in actions}
            print("transition is initialized.")
        else:
            self.trans = trans

    # Define a function that assigns the transitions.
    def add_trans(self, state, action, next_state):
        if state not in self.states or action not in self.actions or next_state not in self.states:
            print("Oops!  That was no valid input.1  Try again...")
            return
        else:
            self.trans[state][action] = next_state
        return

    # Define a function that gets the transitions.
    def get_trans(self, state, action):
        if state not in self.states or action not in self.actions or state not in self.trans or action not in \
                self.trans[state]:
            print("Oops!  That was no valid input.2  Try again...")
            return
        else:
            return self.trans[state][action]

    def get_trans_seq(self, state, input_seq):
        """
        Simulates a transition system.

        Args:
        state: The initial state.
        input_seq: A sequence of symbols (inputs) to process.
        Returns:
        The state reached after processing the sequence of symbols.
        """
        # try implementing this using a recursive function.
        if len(input_seq) == 1:
            return self.get_trans(state, input_seq[0])
        else:
            new_seq = input_seq[1:]
            return self.get_trans_seq(self.get_trans_seq(state, input_seq[:1]), new_seq)

    def get_path(self, state, input_seq):
        """
        :param
        state: the state to start from
        input_seq: a sequence of inputs, for example [a0,a1,a2, ...]
        :return: The sequence of states (a list) starting from the state following the input sequence, if feasible. Otherwise, print ("infeasible action sequence")
        """
        path = [state]
        current_state = state
        for action in input_seq:
            next_state = self.get_trans(current_state, action)
            if next_state is None:
                print("Infeasible action sequence")
                return None
            path.append(next_state)
            current_state = next_state
        return path

    def is_reachable(self, s, ns, visited=None):
        """
        :param s: a state
        :param ns:  another state
        :return: True if there is a path from state s to ns. Otherwise False.
        """
        if visited is None:
            visited = set()
        if s == ns:
            return True
        visited.add(s)
        for action in self.actions:
            next_state = self.get_trans(s, action)
            if next_state is not None and next_state not in visited:
                if self.is_reachable(next_state, ns, visited):
                    return True
        return False


states_1 = {'q0', 'q1', 'q2'}
alphabet_1 = {'o', 'g', 'n',} # The alphabet consists of atomic propositions, the transitions are labeled with combinations
initial_state_1 = 'q1'

transitions_simplified_1 = {
        'q0': {'o': 'q0', 'g': 'q0', 'n': 'q0'},
        'q1': {'n': 'q1', 'g': 'q0', 'o': 'q2'},
        'q2': {'o': 'q2', 'g': 'q2', 'n': 'q2'}
}

# Accepting states based on the Inf(0) condition. State 0 has {0} in the HOA, indicating it's an accepting state for the Buchi condition.
accepting_states_1 = {'q0'}

# Create the DFA object
hoa_dfa_1 = DFA(
    states=states_1,
    input_symbols=alphabet_1, # Using the simplified alphabet including complex labels
    transitions=transitions_simplified_1,
    initial_state=initial_state_1,
    final_states=accepting_states_1 # For Buchi, these are the states that must be visited infinitely often
)

def L(node, gridworld):
    """
    A labeling function that assigns atomic propositions to each state.
    :param state: The current state of the transition system.
    :return: A list of atomic propositions associated with the state.
    """
    if node in gridworld.pit:
        AP = 'o'
    elif node in gridworld.goal:
        AP = 'g'
    else:
        AP = 'n'
    return AP
def dict_to_transys(trans_dict):
    """
    Convert a dictionary-like format back to a TranSys object.
    
    :param trans_dict: A dictionary containing 'states', 'actions', 'initial_state', and 'transitions'
    :return: An instance of TranSys
    """
    # Extract information from the dictionary
    states = list(trans_dict.get('states', []))
    actions = trans_dict.get('actions', [])
    initial_state = trans_dict.get('initial_state', None)
    transitions = trans_dict.get('transitions', {})

    # Create a TranSys object
    trans_sys = TranSys(states, actions, initial_state)

    # Add transitions to the TranSys object
    for (state, action), next_state in transitions.items():
        if isinstance(next_state, list):  # Handle list format of next_state
            next_state = next_state[0]  # Convert back to single state
        trans_sys.add_trans(state, action, next_state)

    return trans_sys

def productMDP(GridWorldMDP, L, aut, acc_cond = 'cosafe'):
    """
    :param gridworldmdp: gridworld mdp
    :param aut: An deterministic automaton.
    :param L: A labeling function that maps a state to a symbol in the automaton's alphabet.
    :param acc_cond: An acceptance condition in the automaton, by default set to 'cosafe'. other options: safe, buchi, co-buchi.
    :return:
    prodTS: The product between the transys and the automaton given the labeling function.
    F: The set of accepting states.
    acc_cond: the acceptance condition.
    """
    
     
    dfa_dict = {
        'states': aut.states,
        'actions': aut.input_symbols,
        'initial_state': aut.initial_state,
        'transitions': {
        (state, action): next_state
        for state, trans in aut.transitions.items()
        for action, next_state in trans.items()
        }
    }
    

    aut_trans = dict_to_transys(dfa_dict)

    product_states = set((s,q) for s in GridWorldMDP.states for q in aut_trans.states)
    product_initial = []

    #for state in GridWorldMDP.states:
        #product_initial.append((state, aut_trans.get_trans(aut_trans.initial_state, L(state))))
    #assuming we start at 0,0 every time to make it easier
    product_inital = [((0,0),aut_trans.get_trans(aut_trans.initial_state, L((0,0), GridWorldMDP))) ]
    product_actions = GridWorldMDP.actions

    F = []
    
    for (s, q) in product_states:
        # Determine if (next_s, next_q) is an accepting state
        if acc_cond == 'cosafe':
            # if q != 'q3' and q!= 'q1' and q!= 'q2' and prodTS.is_reachable((0, 'q1'), (s, q)):
            #     F.add((s, q))
            if q in aut.final_states or (s in GridWorldMDP.goal and q != "q2") :
                F.append((s, q))
        elif acc_cond == 'safe':
            # In 'safe', all states are considered accepting except those in a set of unsafe states.
            if q in aut.final_states or (q == 'q1' and s not in GridWorldMDP.pit) or (s in GridWorldMDP.goal and q != "q2"):
                F.append((s, q))
    
    print("This is the winning region with the automata:", sorted(F))
    print("This includes many states as all states when you are in automata state 0 are valid as that will be after the goal is reached.")
    for st in F:
        if st[1] != 'q0':
            print(st)
    print(len(F))
    transitions = create_transitions(product_states, product_actions, L, GridWorldMDP.p_success, GridWorldMDP.rewards, aut_trans, GridWorldMDP)


    policy = find_policy(GridWorldMDP.N, product_actions, product_states, transitions, GridWorldMDP.gamma)

    return policy, aut_trans
    #print(transitions)
    # i guess from here I can just generate the probabilties of stuff maybe

    # need to redefine the transition function
    
    
def transitions(state, action, actions, p_success, L, rewards, aut_trans, GridWorldMDP):
    """
        Return list of (prob, next_state, reward) tuples
        capturing stochastic movement.
    """
    outcomes = []
    for a_alt in actions:
        prob = p_success if a_alt == action else (1 - p_success) / 3
        s_next = next_state(state, a_alt, L, aut_trans, GridWorldMDP)
        #print("S_next = ",s_next)
        r = rewards[s_next[0]]
        outcomes.append((prob, s_next, r))
        #print("Outcomes = ", outcomes)
    return outcomes
def in_bounds(state, N):
    i, j = state
    return 0 <= i < N and 0 <= j < N
def next_state(state, action, L, aut_trans, GridWorldMDP):
    """Return deterministic next state for given action"""
    i, j = state[0]
    if action == 'U': i -= 1
    elif action == 'D': i += 1
    elif action == 'L': j -= 1
    elif action == 'R': j += 1
    next_s = ((i, j), aut_trans.get_trans(state[1], L((i,j), GridWorldMDP)))
    return next_s if in_bounds(next_s[0], GridWorldMDP.N) else state
def create_transitions(states, actions, L, p_success, rewards, aut_trans, GridWorldMDP):

    # get probability of next state and transition to the next with label
    # need to make transition probabilties

    transition_probabilities = {}
    i = 0
    # need to generate the transitions
    for state in states:
        for action in actions:
            outcomes = transitions(state, action, actions, p_success, L, rewards, aut_trans, GridWorldMDP);
            #print(outcomes)
            for outcome in outcomes:
                #for elem in outcome:
                    #print(elem, ";")
                # insert a new transition probability.
                n_s = outcome[1]
                prob = outcome[0]
                reward = outcome[2]
                #if(state == (0,0)):
                    #print("0,0= ",outcome[1], action, outcome[0])
                # the transition probabilites also include the reward value as this makes it possible to get the policy
                if((state,action, n_s) in transition_probabilities):
                    
                    new = (transition_probabilities[(state, action, n_s)][0] + prob, transition_probabilities[(state, action, n_s)][1] + reward)
                    #print(transition_probabilities[(state, action, n_s)], new)

                    transition_probabilities[(state, action, n_s)] = new
                    #print(new)
                else:
                    #print((prob,reward))
                    transition_probabilities[(state, action, n_s)] =  (prob, reward)
                i += 1
    #print(i)

    #print(transition_probabilities)
    
    return transition_probabilities
def find_policy(N, actions, states, transition_probs, gamma):
        #num_states = N^2
        num_states = len(states)
        num_actions = len(actions)

        c = np.ones(num_states)

        A_ub = []
        b_ub = []

        
        for s_idx, state in enumerate(states):
            for a_idx, action in enumerate(actions):
                row = np.zeros(num_states)
                row[s_idx] = -1

            #print("I am in the idx loop")

            # Coefficients for the next states' values (V(s'))
            immed = 0.0
            for ns_idx, next_state in enumerate(states):
                if (state, action, next_state) in transition_probs:
                    prob = transition_probs[(state,action,next_state)][0]
                    row[ns_idx] += gamma * prob
                    immed += transition_probs[(state,action,next_state)][1]
                    
            A_ub.append(row)
            b_ub.append(-immed)


            # The right-hand side of the inequality is the immediate reward -R(s, a)
            # since the inequality is A_ub @ x <= b_ub
            # We use the state-action reward calculated previously
            
        
        A_ub = np.array(A_ub)
        # b_ub = np.array(self.rewards)
        #b_ub = np.array(self.rewards).flatten()
        b_ub = np.array(b_ub)
        bounds = [(None, None)] * num_states

        #print("Objective function coefficients (c):", c)
        #print("Inequality constraints matrix (A_ub):", A_ub)
        #print("Inequality constraints vector (b_ub):", b_ub)
        #print("Bounds for state values:", bounds)   

        # Solve the linear program
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        # Check if the solver was successful
        if result.success:
            print("Linear programming solved successfully.")
            # The optimal value function is stored in result.x
            optimal_value_function_lp = {}
            for i, state in enumerate(states):
                optimal_value_function_lp[state] = result.x[i]

            #optimal_value_function_lp = {states.at(i): result.x[i] for i in range(num_states)}
            #print("Optimal Value Function (Linear Programming):", optimal_value_function_lp)
        else:
            print("Linear programming did not converge.")
            print("Result:", result)


        optimal_policy_lp = {}

        for state in states:
            best_action = None
            best_q_value = -float('inf')

            for action in actions:
                q_value = 0
                for next_state in states:
                    if (state, action, next_state) in transition_probs:
                        prob = transition_probs[(state, action, next_state)][0]
                        reward = transition_probs[(state, action, next_state)][1]
                        q_value += prob * (reward + gamma * optimal_value_function_lp[next_state])
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action

            optimal_policy_lp[state] = best_action

        #print("Optimal Policy (Linear Programming):")
        #for sta in optimal_policy_lp:
            #print(sta, optimal_policy_lp[sta])
        

        return optimal_policy_lp
def get_path(length, initial_state, policy, p_success, GridWorldMDP, L, aut_trans):
        path = [(initial_state)]
        curr_state = initial_state
        total_reward = 0.0
        for i in range(length):
            # probabilistic next state
            act = policy[curr_state]
            i,j = curr_state[0]
            random_float = random.random();
            if act == 'U':
                if random_float <= p_success :
                    i -= 1
                # biggest num
                elif random_float > p_success and random_float > p_success + 2.0*(1.0-p_success)/3.0 :
                    i += 1
                elif random_float > p_success and random_float >p_success + (1.0-p_success)/3.0 :
                    j -= 1
                elif random_float > p_success :
                    j += 1
            elif act == 'D':
                if random_float <= p_success:
                    i += 1
                elif random_float > p_success and random_float > p_success + 2.0*(1-p_success)/3.0 :
                    i -= 1
                elif random_float > p_success and random_float > p_success + (1.0-p_success)/3.0 :
                    j -= 1
                elif random_float > p_success :
                    j += 1
            elif act == 'L':
                if random_float <= p_success:
                    j -= 1
                elif random_float > p_success and random_float > p_success + 2.0*(1.0-p_success)/3.0 :
                    i += 1
                elif random_float > p_success and random_float > p_success + (1.0-p_success)/3.0 :
                    i -= 1
                elif random_float > p_success :
                    j += 1
            elif act == 'R':
                if random_float <= p_success:
                    j += 1
                elif random_float > p_success and random_float > p_success + 2.0*(1.0-p_success)/3.0 :
                    i -= 1
                elif random_float > p_success and random_float > p_success + (1.0-p_success)/3.0 :
                    j -= 1
                elif random_float > p_success :
                    i += 1

            #gives a random chance for each action to givea different action.
            if in_bounds((i,j), GridWorldMDP.N):
                state = (i,j), aut_trans.get_trans(curr_state[1], L((i,j), GridWorldMDP))
            else:
                state = curr_state
            total_reward += GridWorldMDP.rewards[state[0][0]][state[0][1]]
            path.append((state, GridWorldMDP.rewards[state[0][0]][state[0][1]]))
            curr_state = state
        
        return path, total_reward





     
    


