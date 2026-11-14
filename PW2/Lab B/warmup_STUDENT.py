"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup_STUDENT.py
"""
import numpy as np
from scipy.optimize import newton, minimize

# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

print("=== PART 2A (f(x), start x0 = 0) ===")
# 1. Gradient descent
x_gd = 0.0
lr = 0.1
for _ in range(1000):
    step = lr * df(x_gd)
    x_gd -= step
    if abs(step) < 1e-8:
        break
print(f"1) Gradient Descent: x = {x_gd:.6f}, f(x) = {f(x_gd):.6f}")

# 2. Newton
x_newton = newton(df, 0.0, fprime=d2f)
print(f"2) Newton:           x = {x_newton:.6f}, f(x) = {f(x_newton):.6f}")

# 3. SLSQP
res_slsqp = minimize(f, 0.0, method="SLSQP")
x_slsqp = res_slsqp.x[0]
print(f"3) SLSQP:            x = {x_slsqp:.6f}, f(x) = {f(x_slsqp):.6f}\n")


# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

for x0 in [0.0, 2.0]:
    print(f"=== PART 2B (g(x), start x0 = {x0}) ===")
    
    # 1. Gradient descent
    x_gd = x0
    lr = 0.05
    for _ in range(2000):
        step = lr * dg(x_gd)
        x_gd -= step
        if abs(step) < 1e-8:
            break
    print(f"1) Gradient Descent: x = {x_gd:.6f}, g(x) = {g(x_gd):.6f}")

    # 2. Newton + check second derivative
    x_newt = newton(dg, x0, fprime=d2g)
    sec_deriv = d2g(x_newt)
    point_type = "minimum" if sec_deriv > 0 else "MAXIMUM"
    print(f"2) Newton:           x = {x_newt:.6f}, g(x) = {g(x_newt):.6f} (d2g = {sec_deriv:.2f} -> {point_type})")

    # 3. SLSQP
    res_s = minimize(g, x0, method="SLSQP")
    x_s = res_s.x[0]
    print(f"3) SLSQP:            x = {x_s:.6f}, g(x) = {g(x_s):.6f}\n")