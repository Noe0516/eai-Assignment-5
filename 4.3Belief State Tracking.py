"""
Problem 4.3: Belief State Tracking
Warehouse package location using Bayesian belief updates.

Scanner characteristics:
  P(beep | package here) = 0.90  (true positive rate)
  P(beep | no package)   = 0.05  (false positive rate)
"""

ROOMS = ['A', 'B', 'C', 'D']


# Core Bayesian update

def update_beliefs(beliefs, room_visited, beeped, p_tp=0.90, p_fp=0.05):
    """
    Perform a single Bayesian belief update.

    Args:
        beliefs      : dict mapping room -> probability (must sum to 1)
        room_visited : which room the robot just scanned
        beeped       : True if the scanner beeped, False otherwise
        p_tp         : P(beep | package in this room)  — true positive rate
        p_fp         : P(beep | package NOT here)      — false positive rate

    Returns:
        Updated belief dict (normalized, sums to 1)
    """
    unnormalized = {}
    for loc, prior in beliefs.items():
        if loc == room_visited:
            likelihood = p_tp if beeped else (1 - p_tp)
        else:
            likelihood = p_fp if beeped else (1 - p_fp)
        unnormalized[loc] = likelihood * prior

    normalizing_constant = sum(unnormalized.values())
    return {loc: val / normalizing_constant for loc, val in unnormalized.items()}


def print_beliefs(beliefs, label=""):
    if label:
        print(f"\n{label}")
    for room, prob in sorted(beliefs.items()):
        print(f"  Room {room}: {prob:.4f}")


# Part 1: Room A, no beep

print("PART 1: Room A — no beep")

beliefs = {room: 0.25 for room in ROOMS}
print_beliefs(beliefs, "Prior (uniform):")

beliefs = update_beliefs(beliefs, room_visited='A', beeped=False)
print_beliefs(beliefs, "Posterior after no beep in Room A:")


# Part 2: Room B, beep

print("\nPART 2: Room B — beep")

beliefs = update_beliefs(beliefs, room_visited='B', beeped=True)
print_beliefs(beliefs, "Posterior after beep in Room B:")


# Part 3: Worst-case visits to reach 99% confidence

print("\nPART 3: Worst-case visits for 99% confidence")

def worst_case_visits(target='D', threshold=0.99, p_tp=0.90, p_fp=0.05):
    """
    Simulate the worst case: package is in `target` room,
    robot visits A -> B -> C -> D -> D -> ... until confident.

    The robot receives the most likely observation at each step:
      - no beep when visiting a room without the package
      - beep when visiting the room with the package
    """
    beliefs = {room: 0.25 for room in ROOMS}
    visit_order = ['A', 'B', 'C', 'D', 'D', 'D']

    print(f"\nPackage is secretly in Room {target}. Visiting: {' -> '.join(visit_order)}")

    for visit_num, room in enumerate(visit_order, start=1):
        beeped = (room == target)
        beliefs = update_beliefs(beliefs, room, beeped, p_tp, p_fp)

        obs_str = "BEEP" if beeped else "no beep"
        top_room = max(beliefs, key=beliefs.get)
        top_conf = beliefs[top_room]
        print(f"\nVisit {visit_num}: Room {room} -> {obs_str}")
        print_beliefs(beliefs)
        print(f"  Best guess: Room {top_room} ({top_conf:.4f})")

        if top_conf >= threshold:
            print(f"\n  Reached {threshold*100:.0f}% confidence after {visit_num} visits.")
            return visit_num, beliefs

    return len(visit_order), beliefs

n_visits, final_beliefs = worst_case_visits()
print(f"\nAnswer: {n_visits} visits needed in the worst case.")


# Part 4: Perfect scanner

print("\nPART 4: Perfect scanner (p_tp=1.0, p_fp=0.0)")

beliefs_perfect = {room: 0.25 for room in ROOMS}
print_beliefs(beliefs_perfect, "Prior:")

beliefs_perfect = update_beliefs(beliefs_perfect, 'A', beeped=False, p_tp=1.0, p_fp=0.0)
print_beliefs(beliefs_perfect, "After no beep in Room A:")

beliefs_perfect = update_beliefs(beliefs_perfect, 'B', beeped=True, p_tp=1.0, p_fp=0.0)
print_beliefs(beliefs_perfect, "After beep in Room B:")

# Connection to the logical agent (Chapter 3):
# With a perfect scanner, Bayesian updates reduce to logical elimination.
# A no-beep sets P(package in this room) = 0, ruling it out entirely.
# A beep sets P(package in this room) = 1, identifying the location with certainty.
# This mirrors the knowledge-based agent from Chapter 3: each percept narrows
# the set of possible worlds until exactly one remains. Probabilistic reasoning
# with 0/1 likelihoods IS logical deduction.