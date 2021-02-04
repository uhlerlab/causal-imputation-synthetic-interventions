from time import time
from cmapPy.pandasGEXpress.parse import parse
from filenames import LINCS2_EPSILON_FILE_GCTX, LINCS2_EPSILON_FILE, _format_cmap
from filenames import LINCS3_FILE_GCTX, LINCS3_FILE

LEVEL2 = False
if LEVEL2:
    print('=========================================================')
    print("[processing/convert_to_pickle] Loading Level 2 (original)")
    start = time()
    data = parse(LINCS2_EPSILON_FILE_GCTX).data_df
    data = _format_cmap(data)
    print(f"[processing/convert_to_pickle] Loading/processing took {time() - start} seconds")
    start = time()
    data.to_pickle(LINCS2_EPSILON_FILE)
    print(f"[processing/convert_to_pickle] Saving took {time() - start} seconds")

LEVEL3 = True
if LEVEL3:
    print('=========================================================')
    print("[processing/convert_to_pickle] Loading Level 2 (original)")
    start = time()
    data = parse(LINCS3_FILE_GCTX).data_df
    data = _format_cmap(data)
    print(f"[processing/convert_to_pickle] Loading/processing took {time() - start} seconds")
    start = time()
    data.to_pickle(LINCS3_FILE)
    print(f"[processing/convert_to_pickle] Saving took {time() - start} seconds")
