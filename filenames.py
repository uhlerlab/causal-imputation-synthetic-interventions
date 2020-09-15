import os
import pandas as pd
import numpy as np
from cmapPy.pandasGEXpress.parse import parse
from cmapPy.pandasGEXpress.GCToo import GCToo
from cmapPy.pandasGEXpress.write_gctx import write
from time import time

ROOT_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'raw')
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, 'processed')

LINCS2_EPSILON_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx')
LINCS2_EPSILON_IMPUTED_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978_imputed.gctx')
LINCS2_EPSILON_825_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978_825genes.gctx')

LINCS3_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx')
LINCS3_PRUNED_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328_pruned.gctx')

CELL_INFO_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_cell_info.txt')
PERT_INFO_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_pert_info.txt')
GENE_INFO_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_gene_info.txt')
INST_INFO_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_inst_info.txt')
INST_INFO_EPSILON_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_inst_info_epsilon.txt')

PERT_ID_FIELD = 'pert_iname'


def load_pert_info():
    return pd.read_csv(PERT_INFO_FILE, sep='\t')


def load_inst_info_original():
    return pd.read_csv(INST_INFO_FILE, sep='\t')


def load_inst_info():
    return pd.read_csv(INST_INFO_EPSILON_FILE, sep='\t', index_col=0)


def load_gene_info():
    return pd.read_csv(GENE_INFO_FILE, sep='\t', index_col=0)


def load_cmap():
    return parse(LINCS2_EPSILON_IMPUTED_FILE).data_df.astype(np.uint16)


def load_cmap_filtered():
    return parse(LINCS2_EPSILON_825_FILE).data_df.astype(np.uint16)


def load_cmap_original():
    return parse(LINCS2_EPSILON_FILE).data_df.astype(np.uint16)


def load_cmap_level3():
    if os.path.exists(LINCS3_PRUNED_FILE):
        return parse(LINCS3_PRUNED_FILE).data_df
    else:
        gene_info = load_gene_info()
        l1000_genes = set(map(str, gene_info[gene_info['pr_is_lm'] == 1].index))

        print("Loading")
        start = time()
        rows = parse(LINCS3_FILE, row_meta_only=True)
        row_ixs = rows.index.isin(l1000_genes).nonzero()[0]
        data = parse(LINCS3_FILE, ridx=row_ixs).data_df
        print(f"Loading took {time() - start} seconds")
        print(data.shape)

        print("Saving")
        start = time()
        lincs3_pruned_cmap = GCToo(data)
        write(lincs3_pruned_cmap, LINCS3_PRUNED_FILE)
        print(f"Saving took {time() - start} seconds")

        return data
