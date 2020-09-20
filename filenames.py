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

NUM_DROPOUTS_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'epsilon_num_dropouts.pkl')
CELL_RANK_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'cell_ranks.csv')
PERT_RANK_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'pert_ranks.csv')

PERT_ID_FIELD = 'pert_id'
PERT_OTHER_FIELD = 'pert_iname'


def load_pert_info():
    return pd.read_csv(PERT_INFO_FILE, sep='\t')


def load_inst_info_original():
    return pd.read_csv(INST_INFO_FILE, sep='\t')


def load_inst_info_epsilon():
    return pd.read_csv(INST_INFO_EPSILON_FILE, sep='\t', index_col=0)


def load_gene_info():
    return pd.read_csv(GENE_INFO_FILE, sep='\t', index_col=0)


def _format_cmap(data):
    inst_info = load_inst_info_epsilon()
    data = data.T
    data.index.rename('inst_id', inplace=True)

    # make sure inst_info has same order as data
    data = data.filter(set(inst_info.index), axis=0)
    inst_info = inst_info.loc[data.index]

    # add cell_id and pert_id fields and set as index
    data['cell_id'] = inst_info['cell_id'].values
    data[PERT_ID_FIELD] = inst_info[PERT_ID_FIELD].values
    data = data.set_index(['cell_id', 'pert_id'], append=True)

    return data


def load_cmap_imputed():
    print("Loading Level 2 (imputed)")
    start = time()
    data = parse(LINCS2_EPSILON_IMPUTED_FILE).data_df
    data = _format_cmap(data)
    print(f"Loading/processing took {time() - start} seconds")
    return data


def load_cmap_filtered():
    print("Loading Level 2 (filtered)")
    start = time()
    data = parse(LINCS2_EPSILON_825_FILE).data_df
    data = _format_cmap(data)
    print(f"Loading/processing took {time() - start} seconds")
    return data


def load_cmap_original():
    print("Loading Level 2 (original)")
    start = time()
    data = parse(LINCS2_EPSILON_FILE).data_df
    data = _format_cmap(data)
    print(f"Loading/processing took {time() - start} seconds")
    return data


def load_cmap_level3():
    print("Loading Level 3")
    if os.path.exists(LINCS3_PRUNED_FILE):
        data = parse(LINCS3_PRUNED_FILE).data_df
    else:
        gene_info = load_gene_info()
        l1000_genes = set(map(str, gene_info[gene_info['pr_is_lm'] == 1].index))

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

    data = _format_cmap(data)
    return data


def save_gctx(df, file):
    drop_levels = ['cell_id', PERT_ID_FIELD]
    drop_levels = [level for level in drop_levels if level in df.index.names]
    df.reset_index(drop_levels, drop=True, inplace=True)
    gctoo = GCToo(df.T)
    write(gctoo, file)


def load_num_dropouts():
    return pd.read_pickle(NUM_DROPOUTS_FILE)


def load_pert_ranks():
    return pd.read_csv(PERT_RANK_FILE, index_col=0)


def load_cell_ranks():
    return pd.read_csv(CELL_RANK_FILE, index_col=0)
