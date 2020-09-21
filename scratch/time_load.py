from filenames import load_cmap_original
from time import time
import pandas as pd

start = time()
cm = load_cmap_original()
print(time() - start)

start = time()
cm.to_pickle('scratch/test.pkl')
print(time() - start)

start = time()
c = pd.read_pickle('scratch/test.pkl')
print(time() - start)
