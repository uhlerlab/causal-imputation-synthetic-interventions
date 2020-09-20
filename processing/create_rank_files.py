from filenames import load_inst_info_epsilon, PERT_ID_FIELD, CELL_RANK_FILE, PERT_RANK_FILE
import os

print(f'[create_rank_files] sorting most common celltypes/perturbations')
inst_info_orig = load_inst_info_epsilon()
cells_with_dmso = set(inst_info_orig.query('pert_id == "DMSO"')['cell_id'])
inst_info = inst_info_orig[inst_info_orig['pert_type'] == 'trt_cp']
cell_ranks = inst_info.groupby('cell_id')[PERT_ID_FIELD].nunique().sort_values(ascending=False)
cell_ranks = cell_ranks.filter(cells_with_dmso)
pert_ranks = inst_info.groupby(PERT_ID_FIELD)['cell_id'].nunique().sort_values(ascending=False)

print(f'[create_rank_files] saving sorted lists into files')
os.makedirs('processing/helper_data', exist_ok=True)
cell_ranks.to_csv(CELL_RANK_FILE)
pert_ranks.to_csv(PERT_RANK_FILE)
