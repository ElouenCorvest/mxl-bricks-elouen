from solver import odes
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

y0 = [0, 0, 7.8, 0, 0.1, 0, 0, 0, 0, 0, 1.5, 0.04, 0.04]

res = None
times = None
for t_dur, par in zip([[0,1], [1, 20*60], [20*60, 25*60]], [0, 100, 0]):
    print(t_dur, par)
    sol = solve_ivp(odes, t_dur, y0, args=(par,), dense_output=True, rtol=1e-2, atol=1e-3)
    if res is None:
        res = sol.y
        times = sol.t
        
    else:
        print(res)
        for var_idx in range(len(res)):
            np.append(res[var_idx], sol.y[var_idx])
        np.append(times, sol.t)

QAm, PQH2, pH_lumen, Dy, K_lumen, PC_ox, P700_ox, Z, singletO2, Fd_red, NADPH_pool, Cl_lumen, Cl_stroma = res

plt.plot(times, 1-QAm)

plt.show()