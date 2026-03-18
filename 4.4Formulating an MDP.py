
# ----------------------------
# GRID / MDP SETUP
# ----------------------------

# Grid: (x,y) with (1,1) bottom-left
states = [(x, y) for x in range(1, 5) for y in range(1, 4)]

wall = (2, 2)
goal = (4, 3)
hazard = (4, 2)

# Remove wall
states = [s for s in states if s != wall]

actions = ["N", "S", "E", "W"]
gamma = 1.0

def is_terminal(s):
    return s == goal or s == hazard

def reward(s):
    if s == goal:
        return 1
    elif s == hazard:
        return -1
    else:
        return 0

# ----------------------------
# TRANSITIONS
# ----------------------------

def move(s, a):
    x, y = s
    if a == "N":
        ns = (x, y + 1)
    elif a == "S":
        ns = (x, y - 1)
    elif a == "E":
        ns = (x + 1, y)
    elif a == "W":
        ns = (x - 1, y)

    # If off-grid or into wall → stay
    if ns not in states:
        return s
    return ns

def get_transitions(s, a):
    if is_terminal(s):
        return [(s, 1.0)]

    if a == "N":
        dirs = ["N", "W", "E"]
    elif a == "S":
        dirs = ["S", "E", "W"]
    elif a == "E":
        dirs = ["E", "N", "S"]
    elif a == "W":
        dirs = ["W", "S", "N"]

    probs = [0.8, 0.1, 0.1]

    result = {}
    for d, p in zip(dirs, probs):
        ns = move(s, d)
        result[ns] = result.get(ns, 0) + p

    return list(result.items())

# ----------------------------
# VALUE ITERATION
# ----------------------------

def value_iteration_step(V):
    newV = V.copy()
    for s in states:
        if is_terminal(s):
            continue

        action_values = []
        for a in actions:
            val = 0
            for ns, p in get_transitions(s, a):
                val += p * V[ns]
            action_values.append(val)

        newV[s] = reward(s) + gamma * max(action_values)

    return newV

# Initialize V
V0 = {s: 0 for s in states}
V0[goal] = 1
V0[hazard] = -1

# Run iterations
V1 = value_iteration_step(V0)
V2 = value_iteration_step(V1)

# ----------------------------
# PRINT VALUES
# ----------------------------

def print_values(V, title):
    print(f"\n{title}")
    for y in reversed(range(1, 4)):
        row = []
        for x in range(1, 5):
            s = (x, y)
            if s == wall:
                row.append("WALL".ljust(8))
            else:
                row.append(f"{V[s]:.3f}".ljust(8))
        print(" ".join(row))

print_values(V0, "V0")
print_values(V1, "V1")
print_values(V2, "V2")

# ----------------------------
# EXTRACT POLICY
# ----------------------------

arrow = {
    "N": "↑",
    "S": "↓",
    "E": "→",
    "W": "←"
}

def best_action(V, s):
    if is_terminal(s):
        return "T"

    best_a = None
    best_val = -float("inf")

    for a in actions:
        val = 0
        for ns, p in get_transitions(s, a):
            val += p * V[ns]

        if val > best_val:
            best_val = val
            best_a = a

    return best_a

def print_policy(V):
    print("\nPolicy:")
    for y in reversed(range(1, 4)):
        row = []
        for x in range(1, 5):
            s = (x, y)
            if s == wall:
                row.append("WALL".ljust(6))
            elif s == goal:
                row.append("GOAL".ljust(6))
            elif s == hazard:
                row.append("HAZD".ljust(6))
            else:
                a = best_action(V, s)
                row.append(arrow[a].ljust(6))
        print(" ".join(row))

print_policy(V2)
## This problem helped me understand how to model decision-making under uncertainty using a Markov Decision Process (MDP). By defining the states, actions, transition probabilities, and rewards, I was able to see how a robot can make optimal decisions even when outcomes are not deterministic. Setting up the grid and identifying the wall, goal, and hazard made it clear how constraints and risks affect possible movements.
#Using value iteration in Python made the process more concrete. At first, all non-terminal states had a value of zero, but after each iteration, the values began to propagate outward from the terminal states. I noticed that states closer to the goal quickly gained higher values, while states near the hazard either stayed low or were influenced negatively. By the second iteration, I could already see how the algorithm was prioritizing safer paths toward the goal.

#The policy output was especially interesting because it did not always match the shortest path. Instead, it avoided the hazard by choosing safer actions, even if that meant taking a longer route. This showed me how uncertainty in transitions (like the 10% drift) forces the robot to be more cautious. In particular, states near the hazard required careful decisions since even a small probability of drifting could lead to a large negative reward.

#Overall, this problem reinforced how MDPs balance risk and reward. It also showed how iterative methods like value iteration gradually improve decision-making and lead to an optimal policy. Implementing it in Python helped me better visualize how values update and how the optimal strategy emerges over time.
