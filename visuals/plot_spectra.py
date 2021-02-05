from evaluation.helpers import get_data_block
from numpy.linalg import svd
import matplotlib.pyplot as plt
import seaborn as sns
from filenames import load_cmap_original
from src.matrix import approximate_rank
sns.set()

df, _, _, _ = get_data_block(
    num_cells=None,
    num_perts=100,
    name='level2'
)

# df_orig = load_cmap_original()

vals = df.values
vals -= vals.mean(axis=0)
print(approximate_rank(vals))
u, s, v = svd(vals)
plt.clf()
plt.plot(s)
plt.ylabel("Singular values")
plt.tight_layout()
plt.savefig("visuals/figures/spectra.png")
