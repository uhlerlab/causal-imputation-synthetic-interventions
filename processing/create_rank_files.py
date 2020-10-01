from filenames import load_inst_info_epsilon, PERT_ID_FIELD, CELL_RANK_FILE, PERT_RANK_FILE
from filenames import load_inst_info_most_common, CELL_RANK_COMMON_FILE, PERT_RANK_COMMON_FILE
import os

print(f'[create_rank_files] sorting most common celltypes/perturbations')
inst_info_orig = load_inst_info_epsilon()
cells_with_dmso = set(inst_info_orig.query('pert_id == "DMSO"')['cell_id'])
inst_info = inst_info_orig[inst_info_orig['pert_type'] == 'trt_cp']
inst_info = inst_info.query('cell_id in @cells_with_dmso')
cell_ranks = inst_info.groupby('cell_id')[PERT_ID_FIELD].nunique().sort_values(ascending=False)
pert_ranks = inst_info.groupby(PERT_ID_FIELD)['cell_id'].nunique().sort_values(ascending=False)

inst_info_most_common_orig = load_inst_info_most_common()
cells_with_dmso_common = set(inst_info_most_common_orig.query('pert_id == "DMSO"')['cell_id'])
inst_info_most_common = inst_info_most_common_orig[inst_info_most_common_orig['pert_type'] == 'trt_cp']
inst_info_most_common = inst_info_most_common.query('cell_id in @cells_with_dmso_common')
cell_ranks_most_common = inst_info_most_common.groupby('cell_id')[PERT_ID_FIELD].nunique().sort_values(ascending=False)
pert_ranks_most_common = inst_info_most_common.groupby(PERT_ID_FIELD)['cell_id'].nunique().sort_values(ascending=False)

print(f'[create_rank_files] saving sorted lists into files')
os.makedirs('processing/helper_data', exist_ok=True)
cell_ranks.to_csv(CELL_RANK_FILE)
pert_ranks.to_csv(PERT_RANK_FILE)
cell_ranks_most_common.to_csv(CELL_RANK_COMMON_FILE)
pert_ranks_most_common.to_csv(PERT_RANK_COMMON_FILE)
