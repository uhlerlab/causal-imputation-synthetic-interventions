from filenames import load_gene_info, LINCS3_PRUNED_FILE, LINCS3_FILE_GCTX, _format_cmap
from time import time
from cmapPy.pandasGEXpress.parse import parse

gene_info = load_gene_info()
l1000_genes = set(map(str, gene_info[gene_info['pr_is_lm'] == 1].index))

start = time()
rows = parse(LINCS3_FILE_GCTX, row_meta_only=True)
row_ixs = rows.index.isin(l1000_genes).nonzero()[0]
data = parse(LINCS3_FILE_GCTX, ridx=row_ixs).data_df
print(f"[processing/prune_level3] Loading took {time() - start} seconds")

data_ = _format_cmap(data)

print("[processing/prune_level3] Saving")
start = time()
data_.to_pickle(LINCS3_PRUNED_FILE)
print(f"[processing/prune_level3] Saving took {time() - start} seconds")

