
from filenames import load_cmap_level3_original, LINCS3_PRUNED_FILE, load_gene_info
from cmapPy.pandasGEXpress.GCToo import GCToo
from cmapPy.pandasGEXpress.write_gctx import write
from time import time

print("Loading CMAP level 3 data")
start = time()
lincs3 = load_cmap_level3_original()
print(f"Loading took {time() - start} seconds")
gene_info = load_gene_info()

print("Pruning")
l1000_genes = set(gene_info[gene_info['pr_is_lm'] == 1].index)
lincs3_pruned = lincs3.data_df.filter(l1000_genes, axis=0)
# lincs3_pruned = lincs3_pruned.sort_index()

print("Saving")
start = time()
lincs3_pruned_cmap = GCToo(lincs3_pruned)
write(lincs3_pruned_cmap, LINCS3_PRUNED_FILE)
print(f"Saving took {time() - start} seconds")
