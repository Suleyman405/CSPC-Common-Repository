"""
PW2 Lab B Part 5 (bonus) -- find a titration's equivalence point.

titration.csv holds a titration curve: pH versus the volume of base added.
The equivalence point is the volume where the pH changes fastest (the steep
jump). Numerically, that is where the SLOPE of the pH curve is largest.
Run:  python titration.py
"""
import numpy as np
import matplotlib.pyplot as plt

# TODO 1: read titration.csv (columns volume_base, pH) into arrays V, pH.
data = np.loadtxt('titration.csv', delimiter=',', skiprows=1)
V = data[:, 0]
pH = data[:, 1]

# TODO 2: compute the slope of the pH curve with np.gradient(pH, V), and find
#         the volume where that slope is largest (np.argmax). That is the
#         equivalence point. Print it.
slope = np.gradient(pH, V)
eq_idx = np.argmax(slope)
V_eq = V[eq_idx]
print(f"Equivalence point volume: {V_eq} mL")

# TODO 3: make two plots side by side: (left) pH vs volume with a line at the
#         equivalence point; (right) the slope vs volume, showing it peaks
#         at the equivalence point. Save as titration.png.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: pH curve with equivalence line
ax1.plot(V, pH, 'b-', label='pH curve')
ax1.axvline(V_eq, color='r', linestyle='--', label=f'Eq. Point ({V_eq} mL)')
ax1.set_xlabel('Volume of Base')
ax1.set_ylabel('pH')
ax1.set_title('Titration Curve')
ax1.legend()
ax1.grid(True)

# Right plot: slope peak at equivalence point
ax2.plot(V, slope, 'g-', label='dpH / dV')
ax2.axvline(V_eq, color='r', linestyle='--', label='Eq. Point')
ax2.set_xlabel('Volume of Base')
ax2.set_ylabel('Slope')
ax2.set_title('Derivative (Slope) Curve')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('titration.png')
plt.show()