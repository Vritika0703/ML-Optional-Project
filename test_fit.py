import numpy as np
from scipy.optimize import curve_fit
def power_law(N, a, alpha, c): return a * N**(-alpha) + c
ns = np.array([1_000_000, 3_000_000, 10_000_000, 30_000_000, 88_000_000], dtype=float)
sp = np.array([4.21, 3.87, 3.52, 3.18, 2.89])
mup= np.array([4.18, 3.79, 3.40, 3.02, 2.71])
b = ([0.01, 0.001, 0.0], [100.0, 2.0, 5.0])
p1,_ = curve_fit(power_law, ns, sp, p0=[2.0, 0.08, 2.0], bounds=b, maxfev=20000)
p2,_ = curve_fit(power_law, ns, mup, p0=[2.0, 0.10, 1.8], bounds=b, maxfev=20000)
print(f"SP: a={p1[0]:.2f} alpha={p1[1]:.4f} c={p1[2]:.2f}")
print(f"mup: a={p2[0]:.2f} alpha={p2[1]:.4f} c={p2[2]:.2f}")
print(f"SP(880M): {power_law(880000000, *p1):.2f}")
print(f"mup(880M): {power_law(880000000, *p2):.2f}")
