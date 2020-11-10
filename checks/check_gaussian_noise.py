import numpy as np
import matplotlib.pyplot as plt

import pdb

plen = 5
chunks_p_unit = 10


noise = np.random.normal(0, 2, (plen, chunks_p_unit))
base = np.tile(np.arange(plen), (chunks_p_unit, 1)).T

final = base + noise
final = np.rint(final)


outs = np.unique(final, return_counts=True)

plt.plot(outs[0], outs[1]/chunks_p_unit)
plt.show()