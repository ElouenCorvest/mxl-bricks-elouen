import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

res = pd.read_csv(Path(__file__).parent / "here.csv", names=["QA", "QAm", "PQ", "PQH2", "Hin", "pHlumen", "Dy", "pmf", "deltaGatp", "Klumen", "Kstroma", "ATP_made", "PC_ox", "PC_red", "P700_ox", "P700_red", "Z", "V", "NPQ", "singletO2", "Phi2", "LEF", "Fd_ox", "Fd_red", "ATP_pool", "ADP_pool", "NADPH_pool", "NADP_pool","Cl_lumen", "Cl_stroma", "Hstroma", "pHstroma"])

changes = res.diff()[1:]

fig, ax = plt.subplots()

ax.plot((0.06 * res["pHlumen"] + res["Dy"]))
ax.plot(res["pmf"])

plt.show()
    