import random
from collections import Counter

# Grid dimensions
WIDTH, HEIGHT = 4, 3

# Terminal states and their rewards
GOAL = (4, 3)
HAZARD = (4, 2)
TERMINALS = {GOAL: +1.0, HAZARD: -1.0}

# Living reward
LIVING_REWARD = -0.04

# Wall
WALL = (2, 2)

# States
STATES = [
    (x, y)
    for x in range(1, WIDTH + 1)
    for y in range(1, HEIGHT + 1)
    if (x, y) != WALL
]

# Actions
ACTIONS = {
    "North": (0, 1),
    "South": (0, -1),
    "East": (1, 0),
    "West": (-1, 0),
}

ARROWS = {"North": "↑", "South": "↓", "East": "→", "West": "←"}


def reward(state):
    if state in TERMINALS:
        return TERMINALS[state]
    return LIVING_REWARD


def get_perpendicular(action):
    if action in ("North", "South"):
        return ["West", "East"]
    return ["North", "South"]


def attempt_move(state, action):
    dx, dy = ACTIONS[action]
    nx, ny = state[0] + dx, state[1] + dy
    if 1 <= nx <= WIDTH and 1 <= ny <= HEIGHT and (nx, ny) != WALL:
        return (nx, ny)
    return state


def transitions(state, action):
    if state in TERMINALS:
        return {}

    outcomes = {}

    intended = attempt_move(state, action)
    outcomes[intended] = outcomes.get(intended, 0) + 0.8

    for perp in get_perpendicular(action):
        drifted = attempt_move(state, perp)
        outcomes[drifted] = outcomes.get(drifted, 0) + 0.1

    return outcomes


def value_iteration(gamma=0.99, epsilon=1e-6):
    V = {s: 0.0 for s in STATES}
    iteration = 0

    while True:
        V_new = {}
        delta = 0

        for s in STATES:
            if s in TERMINALS:
                V_new[s] = reward(s)
                continue

            best_value = float("-inf")

            for a in ACTIONS:
                expected = sum(
                    prob * V[s_next]
                    for s_next, prob in transitions(s, a).items()
                )
                best_value = max(best_value, expected)

            V_new[s] = reward(s) + gamma * best_value
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        iteration += 1

        if delta < epsilon:
            break

    return V, iteration


def extract_policy(V, gamma=0.99):
    policy = {}

    for s in STATES:
        if s in TERMINALS:
            policy[s] = None
            continue

        best_action = None
        best_value = float("-inf")

        for a in ACTIONS:
            expected = sum(
                prob * V[s_next]
                for s_next, prob in transitions(s, a).items()
            )
            value = reward(s) + gamma * expected

            if value > best_value:
                best_value = value
                best_action = a

        policy[s] = best_action

    return policy


def simulate_step(state, action):
    dist = transitions(state, action)
    states = list(dist.keys())
    probs = list(dist.values())
    return random.choices(states, weights=probs, k=1)[0]


# ---------------- Part 1: Transition Check ----------------
random.seed(42)
counts = Counter()

for _ in range(10000):
    counts[simulate_step((3, 1), "North")] += 1

print("Empirical transition frequencies from (3,1), action North:")
for s, c in sorted(counts.items()):
    print(f"{s}: {c/10000:.3f}")


# ---------------- Part 2: Episode Simulation ----------------
def run_episode(policy, start=(1, 1), max_steps=100):
    state = start
    trajectory = [state]
    total_reward = reward(state)

    for _ in range(max_steps):
        if state in TERMINALS:
            break

        action = policy[state]
        state = simulate_step(state, action)

        trajectory.append(state)
        total_reward += reward(state)

    if state == GOAL:
        outcome = "goal"
    elif state == HAZARD:
        outcome = "hazard"
    else:
        outcome = "timeout"

    return trajectory, total_reward, outcome


random.seed(42)
V, num_iters = value_iteration()
optimal_policy = extract_policy(V)

print(f"\nValue iteration converged in {num_iters} iterations.")

results = [run_episode(optimal_policy) for _ in range(1000)]
outcomes = [r[2] for r in results]
rewards = [r[1] for r in results]

print(f"Goal reached: {outcomes.count('goal') / 1000:.3f}")
print(f"Hazard hit: {outcomes.count('hazard') / 1000:.3f}")
print(f"Avg reward: {sum(rewards) / 1000:.3f}")


# ---------------- Part 3: Greedy vs Optimal ----------------
def greedy_policy_action(state):
    gx, gy = GOAL
    sx, sy = state

    if sx < gx:
        return "East"
    elif sx > gx:
        return "West"
    elif sy < gy:
        return "North"
    else:
        return "South"


greedy_policy = {s: greedy_policy_action(s) for s in STATES if s not in TERMINALS}
greedy_policy[GOAL] = None
greedy_policy[HAZARD] = None

opt_results = [run_episode(optimal_policy) for _ in range(1000)]
greedy_results = [run_episode(greedy_policy) for _ in range(1000)]

print("\nOptimal policy:")
print(f"Goal: {sum(1 for r in opt_results if r[2]=='goal')/1000:.3f}")
print(f"Hazard: {sum(1 for r in opt_results if r[2]=='hazard')/1000:.3f}")

print("\nGreedy policy:")
print(f"Goal: {sum(1 for r in greedy_results if r[2]=='goal')/1000:.3f}")
print(f"Hazard: {sum(1 for r in greedy_results if r[2]=='hazard')/1000:.3f}")


# ---------------- Part 4: Gamma Experiment ----------------
print("\nDiscount factor experiment:")

for gamma in [0.1, 0.5, 0.9, 0.99]:
    V, _ = value_iteration(gamma)
    policy = extract_policy(V, gamma)

    results = [run_episode(policy) for _ in range(1000)]
    outcomes = [r[2] for r in results]
    rewards = [r[1] for r in results]

    print(f"gamma={gamma:.2f} goal={outcomes.count('goal')/1000:.3f} "
          f"hazard={outcomes.count('hazard')/1000:.3f} "
          f"avg={sum(rewards)/1000:.3f}")


# ---------------- Part 5: Add Second Hazard ----------------
TERMINALS[(2, 3)] = -1.0

V, _ = value_iteration()
policy = extract_policy(V)

print("\nPolicy with second hazard:\n")

for y in range(HEIGHT, 0, -1):
    row = []
    for x in range(1, WIDTH + 1):
        s = (x, y)

        if s == GOAL:
            row.append(" GOAL")
        elif s in TERMINALS and TERMINALS[s] < 0:
            row.append(" HAZD")
        elif s not in STATES:
            row.append(" WALL")
        else:
            row.append(f"  {ARROWS[policy[s]]} ")

    print("".join(row))


# ---------------- Reflection ----------------
# This problem helped me connect the theory behind MDPs to an actual working agent.
# Instead of only solving equations, I was able to simulate how an agent behaves
# in a stochastic environment where actions are not always guaranteed.

# The simulation showed that even with an optimal policy, the agent cannot always
# reach the goal because of randomness in transitions. However, the optimal policy
# still performs well by maximizing long-term reward.

# Comparing the greedy policy to the optimal policy was important. The greedy policy
# ignores risk and moves straight toward the goal, which leads to more frequent
# failures. The optimal policy instead balances risk and reward by avoiding the
# hazard when necessary.

# The discount factor experiment showed how gamma affects decision-making. Lower
# values make the agent short-sighted, while higher values encourage long-term
# planning and safer behavior.

# Adding a second hazard made the environment more complex and showed how sensitive
# policies are to changes in the environment. The agent adapted by becoming more
# conservative.

# Overall, this assignment reinforced how value iteration works and how MDPs allow
# an agent to plan under uncertainty while balancing efficiency and safety.