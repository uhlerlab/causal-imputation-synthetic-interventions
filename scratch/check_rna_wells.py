from filenames import load_inst_info_original
from tqdm import tqdm

inst_info = load_inst_info_original()
inst_info = inst_info[inst_info['pert_type'].isin({'ctl_vehicle', 'trt_cp'})]
rna_wells_to_perts = inst_info.groupby('rna_well')['pert_id'].unique().to_dict()
rna_plates_to_perts = inst_info.groupby('rna_plate')['pert_id'].unique().to_dict()
rna_plate_sizes = inst_info.groupby('rna_plate').size().sort_values()
print(f"Number of wells: {len(rna_wells_to_perts)}")
print(f"Number of plates: {len(rna_plates_to_perts)}")
rna_plates_no_dmso = {well for well, perts in rna_plates_to_perts.items() if 'DMSO' not in perts}
a = inst_info[inst_info['rna_plate'] == 'T2D001']

rna_plates_no_dmso_perts = {
    plate: inst_info[inst_info['rna_plate'] == plate]['pert_id'].unique()
    for plate in rna_plates_no_dmso
}
rna_plate_no_dmso_num_perts = {plate: len(perts) for plate, perts in rna_plates_no_dmso_perts.items()}

rna_plates_no_dmso_cells = {
    plate: inst_info[inst_info['rna_plate'] == plate]['cell_id'].unique()
    for plate in rna_plates_no_dmso
}

# want to check: for each plate, each cell type has a sample of DMSO
plates_without_pair = set()
for plate in tqdm(rna_plate_sizes.index):
    samples = inst_info[inst_info['rna_plate'] == plate]
    cell_ids = set(samples['cell_id'])
    cell_id2perts = samples.groupby('cell_id')['pert_id'].unique().to_dict()
    all_cells_have_dmso = all('DMSO' in perts for perts in cell_id2perts.values())
    if not all_cells_have_dmso:
        plates_without_pair.add(plate)


# 2,052 rna plates
# each plate divided into 5 t0 1432 wells
# some wells have overlapping samples
