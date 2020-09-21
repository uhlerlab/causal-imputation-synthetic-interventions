from time import time
from cmapPy.pandasGEXpress.parse import parse
from filenames import LINCS2_EPSILON_FILE_GCTX, LINCS2_EPSILON_FILE, _format_cmap


print("Loading Level 2 (original)")
start = time()
data = parse(LINCS2_EPSILON_FILE_GCTX).data_df
data = _format_cmap(data)
print(f"Loading/processing took {time() - start} seconds")
start = time()
data.to_pickle(LINCS2_EPSILON_FILE)
print(f"Saving took {time() - start} seconds")
