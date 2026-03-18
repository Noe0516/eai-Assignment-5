
# Prior
P_D = 0.15
P_not_D = 0.85

# Old sensor (CORRECT VALUES)
TP_old = 0.85   # P(C=1 | D=1)
FP_old = 0.08   # P(C=1 | D=0)

# New sensor
TP_new = 0.95
FP_new = 0.02



# Helper: Bayes Rule
# -------------------------------
def bayes(prior, likelihood, evidence):
    return (likelihood * prior) / evidence


# -------------------------------
# Part 1: No Creaking (C=0)
# -------------------------------
P_noC_given_D = 1 - TP_old      # 0.15
P_noC_given_notD = 1 - FP_old   # 0.92

P_noC = (P_noC_given_D * P_D) + (P_noC_given_notD * P_not_D)

P_D_given_noC = bayes(P_D, P_noC_given_D, P_noC)

print("Part 1:")
print(f"P(D | no creak) = {P_D_given_noC:.4f}")  # ~0.028


# -------------------------------
# Part 2: Creaking (C=1)
# -------------------------------
P_C_given_D = TP_old
P_C_given_notD = FP_old

P_C = (P_C_given_D * P_D) + (P_C_given_notD * P_not_D)

P_D_given_C = bayes(P_D, P_C_given_D, P_C)

print("\nPart 2:")
print(f"P(D | creak) = {P_D_given_C:.4f}")  # ~0.652


# -------------------------------
# Part 3: New Sensor
# -------------------------------

# No creak
P_noC_given_D_new = 1 - TP_new
P_noC_given_notD_new = 1 - FP_new

P_noC_new = (P_noC_given_D_new * P_D) + (P_noC_given_notD_new * P_not_D)
P_D_given_noC_new = bayes(P_D, P_noC_given_D_new, P_noC_new)

# Creak
P_C_given_D_new = TP_new
P_C_given_notD_new = FP_new

P_C_new = (P_C_given_D_new * P_D) + (P_C_given_notD_new * P_not_D)
P_D_given_C_new = bayes(P_D, P_C_given_D_new, P_C_new)

print("\nPart 3 (Sensor Comparison):")
print(f"Old sensor:  P(D | no creak) = {P_D_given_noC:.3f}")
print(f"New sensor:  P(D | no creak) = {P_D_given_noC_new:.3f}")
print()
print(f"Old sensor:  P(D | creak) = {P_D_given_C:.3f}")
print(f"New sensor:  P(D | creak) = {P_D_given_C_new:.3f}")


# -------------------------------
# Part 4: Two Adjacent Squares
# -------------------------------

P_D1 = 0.15
P_D2 = 0.15

# P(at least one damaged)
P_at_least_one = 1 - (1 - P_D1) * (1 - P_D2)

# P(no creak | at least one damaged)
P_noC_given_at_least_one = 1 - TP_old

# Total probability of no creak
P_noC_total = (
    P_noC_given_at_least_one * P_at_least_one +
    (1 - FP_old) * (1 - P_at_least_one)
)

# Posterior
P_D1_given_noC = (P_D1 * (1 - TP_old)) / P_noC_total

print("\nPart 4:")
print(f"P(at least one damaged) = {P_at_least_one:.3f}")
print(f"P(D1 | no creak) ≈ {P_D1_given_noC:.3f}")