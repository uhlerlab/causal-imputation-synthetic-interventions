"""
Filter inst_info so that it only contains metadata for samples in the LINCS2_EPSILON file.
"""

from filenames import load_inst_info_original, INST_INFO_EPSILON_FILE, LINCS2_EPSILON_FILE
from cmapPy.pandasGEXpress.parse import parse

data = parse(LINCS2_EPSILON_FILE).data_df
inst_info_original = load_inst_info_original()
inst_info_filtered = inst_info_original[inst_info_original['inst_id'].isin(set(data.columns))]
inst_info_filtered.to_csv(INST_INFO_EPSILON_FILE, sep='\t', index=False)
