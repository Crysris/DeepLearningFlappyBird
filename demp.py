import numpy as np
a = np.zeros((600, 200, 3), dtype=int)
np.save("a.npy", a)