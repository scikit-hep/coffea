import sys

import matplotlib.pyplot as plt
import mplhep

from coffea.util import load

mplhep.style.use("ROOT")

full_result = load("dimuonmass.coffea")
h_name = sys.argv[1]

for dataset, result in full_result.items():
    result[h_name].plot(label=dataset)
plt.legend()
plt.ylabel("N Events")
plt.savefig(h_name + ".png", bbox_inches="tight")
plt.clf()
