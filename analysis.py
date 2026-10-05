"""
PW2 Lab A -- Motion from tracking data.

Read noisy free-fall position measurements, then:
  - differentiate once  -> velocity
  - differentiate twice -> acceleration (should be ~ constant -g, but noisy!)
  - integrate the acceleration back up -> recover velocity and position

Complete the TODOs. Run:  python analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# TODO 1: 
data = np.loadtxt("freefall.csv", delimiter=",", skiprows=1)

t = data[:, 0]
y = data[:, 1]
# TODO 2: 
v = np.gradient(y, t)

a = np.gradient(v, t)

print("Mean acceleration:", np.mean(a))
print("Standard deviation:", np.std(a))

# TODO 3: 
v_recovered = cumulative_trapezoid(a, t, initial=0) + v[0]

y_recovered = cumulative_trapezoid(v_recovered, t, initial=0) + y[0]

difference = np.max(np.abs(y_recovered - y))

print("Largest position difference:", difference)

# TODO 4: 
fig, ax = plt.subplots(3, 1, figsize=(8, 10))

ax[0].plot(t, y, label="Position")
ax[0].plot(t, y_recovered, label="Recovered position")
ax[0].set_ylabel("Position (m)")
ax[0].legend()
ax[0].grid()

ax[1].plot(t, v)
ax[1].set_ylabel("Velocity (m/s)")
ax[1].grid()

ax[2].plot(t, a, label="Acceleration")
ax[2].axhline(-9.81, color="red", linestyle="--", label="True -9.81")
ax[2].set_ylabel("Acceleration (m/s²)")
ax[2].set_xlabel("Time (s)")
ax[2].legend()
ax[2].grid()

plt.tight_layout()
plt.savefig("motion.png")
plt.show()