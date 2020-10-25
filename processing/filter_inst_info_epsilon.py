"""
Filter inst_info so that it only contains metadata for samples in the LINCS2_EPSILON file.
"""

from filenames import load_inst_info_original, INST_INFO_EPSILON_FILE, LINCS2_EPSILON_FILE_GCTX
from cmapPy.pandasGEXpress.parse import parse
from time import time

start = time()
inst_ids = set(parse(LINCS2_EPSILON_FILE_GCTX, col_meta_only=True).index)
inst_info_original = load_inst_info_original()
inst_info_original.query('inst_id in @inst_ids', inplace=True)
inst_info_original.to_pickle(INST_INFO_EPSILON_FILE)
print(f"[processing/filter_inst_info_epsilon] Filtering inst_info took {time() - start} seconds")
