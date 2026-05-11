import numpy as np
from scipy.optimize import curve_fit

def power_law(N, a, alpha, c):
    return a * N**(-alpha) + c

PARAM_COUNTS = [1_000_000, 3_000_000, 10_000_000, 30_000_000, 88_000_000]
DEMO_SP_VAL_LOSSES  = [4.21, 3.87, 3.52, 3.18, 2.89]
DEMO_MUP_VAL_LOSSES = [4.18, 3.79, 3.40, 3.02, 2.71]

ns = np.array(PARAM_COUNTS, dtype=float)
sp = np.array(DEMO_SP_VAL_LOSSES)
mup= np.array(DEMO_MUP_VAL_LOSSES)

bounds = ([0.01, 0.001, 0.5], [100.0, 2.0, 5.0])
sp_popt,_  = curve_fit(power_law, ns, sp, p0=[2.0, 0.08, 2.0], bounds=bounds, maxfev=20000)

bounds2 = ([0.001, 0.001, 0.0], [100.0, 2.0, 5.0])
sp_popt2,_  = curve_fit(power_law, ns, sp, p0=[2.0, 0.08, 2.0], bounds=bounds2, maxfev=20000)
mup_popt2,_ = curve_fit(power_law, ns, mup, p0=[2.0, 0.10, 1.8], bounds=bounds2, maxfev=20000)

print(f"Fig4 SP: a={sp_popt[0]:.2f} alpha={sp_popt[1]:.4f} c={sp_popt[2]:.2f}")
print(f"Fig5 SP: a={sp_popt2[0]:.2f} alpha={sp_popt2[1]:.4f} c={sp_popt2[2]:.2f}")
print(f"Fig5 mup: a={mup_popt2[0]:.2f} alpha={mup_popt2[1]:.4f} c={mup_popt2[2]:.2f}")
