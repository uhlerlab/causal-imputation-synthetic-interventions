import random
from cmapPy.pandasGEXpress.GCToo import GCToo
from cmapPy.pandasGEXpress.write_gctx import write
from filenames import *
import pandas as pd
import os
import ipdb

PERTS_PER_CELLTYPE_FILE = 'processing/helper_data/perts_per_celltype.csv'
CELLTYPES_PER_PERT_FILE = 'processing/helper_data/celltypes_per_pert.csv'


def save_most_common_lists():
    print(f'[save_most_common_list] loading original LINCS2 data')
    data_df = load_cmap()

    print(f'[save_most_common_list] sorting most common celltypes/perturbations')
    inst_info = load_inst_info()
    inst_info = inst_info.filter(set(data_df.columns), axis=0)
    inst_info = inst_info[inst_info['pert_type'] == 'trt_cp']
    perts_per_celltype = inst_info.groupby('cell_id')[PERT_ID_FIELD].nunique().sort_values(ascending=False)
    celltypes_per_pert = inst_info.groupby(PERT_ID_FIELD)['cell_id'].nunique().sort_values(ascending=False)

    print(f'[save_most_common_list] saving sorted lists into files')
    os.makedirs('processing/helper_data', exist_ok=True)
    perts_per_celltype.to_csv(PERTS_PER_CELLTYPE_FILE)
    celltypes_per_pert.to_csv(CELLTYPES_PER_PERT_FILE)


class MostCommonManager:
    def __init__(self, num_celltypes, num_perts):
        """
        Base class for selecting a subset of the dataset containing the most common cell types and perturbations.
        """
        self.nc = num_celltypes
        self.np = num_perts
        if not os.path.exists(PERTS_PER_CELLTYPE_FILE):
            save_most_common_lists()

    def get_most_common_gctx(self, overwrite=False):
        filename = os.path.join('processing', 'helper_data', 'most_common', f'{self.nc}celltypes_{self.np}perts.gctx')
        if overwrite or not os.path.exists(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # get most common celltypes/perts
            perts_per_celltype = pd.read_csv(PERTS_PER_CELLTYPE_FILE, index_col=0)
            selected_celltypes = set(perts_per_celltype.index[:self.nc]) if self.nc is not None else set(perts_per_celltype.index)
            celltypes_per_pert = pd.read_csv(CELLTYPES_PER_PERT_FILE, index_col=0)
            selected_perts = set(celltypes_per_pert.index[:self.np]) | {'DMSO'}

            print(f'[MostCommonManager.get_most_common_gctx] loading original LINCS2 data')
            full_df = load_cmap()

            print(f'[MostCommonManager.get_most_common_gctx] filtering to most common {self.nc} celltypes and {self.np} perturbations')
            inst_info = load_inst_info()
            # inst_info = pd.read_csv(INST_INFO_FILE, sep='\t', index_col=0)
            selected_inst_ids = inst_info[
                inst_info[PERT_ID_FIELD].isin(selected_perts) &
                inst_info['cell_id'].isin(selected_celltypes)
            ].index
            print(f"[MostCommonManager.get_most_common_gctx] {len(selected_inst_ids)} inst_id's selected")
            new_df = full_df.filter(selected_inst_ids)

            print(f'[MostCommonManager.get_most_common_gctx] saving to {filename}')
            gctoo = GCToo(new_df)
            write(gctoo, filename)
        else:
            print(f'[MostCommonManager.get_most_common_gctx] loading from {filename}')
            gctoo = parse(filename)

        df = gctoo.data_df
        return df

    def compute_availability(self):
        pass

