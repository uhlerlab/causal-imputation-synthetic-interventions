import os
import pandas as pd
from time import time

ROOT_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'raw')
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, 'processed')

LINCS2_EPSILON_FILE_GCTX = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx')
LINCS2_EPSILON_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.pkl')
LINCS2_EPSILON_IMPUTED_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978_imputed.pkl')
LINCS2_EPSILON_825_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978_825genes.pkl')

LINCS3_FILE_GCTX = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx')
LINCS3_PRUNED_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328_pruned.pkl')

CELL_INFO_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_cell_info.txt')
PERT_INFO_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_pert_info.txt')
GENE_INFO_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_gene_info.txt')
INST_INFO_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_inst_info.txt')
INST_INFO_EPSILON_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'GSE92742_Broad_LINCS_inst_info_epsilon.pkl')

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
    return pd.read_pickle(INST_INFO_EPSILON_FILE)


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
    data.set_index(['cell_id', 'pert_id'], append=True, inplace=True)

    return data


def load_cmap_imputed():
    print("[load_cmap_imputed] Loading...")
    start = time()
    data = pd.read_pickle(LINCS2_EPSILON_IMPUTED_FILE)
    print(f"[load_cmap_imputed] ... Loading took {time() - start} seconds")
    return data


def load_cmap_filtered():
    print("[load_cmap_filtered] Loading...")
    start = time()
    data = pd.read_pickle(LINCS2_EPSILON_825_FILE)
    print(f"[load_cmap_filtered] ... Loading took {time() - start} seconds")
    return data


def load_cmap_original():
    print("[load_cmap_original] Loading...")
    start = time()
    data = pd.read_pickle(LINCS2_EPSILON_FILE)
    print(f"[load_cmap_original] ... Loading took {time() - start} seconds")
    return data


def load_cmap_level3():
    print("[load_cmap_level3] Loading...")
    start = time()
    data = pd.read_pickle(LINCS3_PRUNED_FILE)
    print(f"[load_cmap_level3] ... Loading took {time() - start} seconds")
    return data


def load_num_dropouts():
    return pd.read_pickle(NUM_DROPOUTS_FILE)


def load_pert_ranks():
    return pd.read_csv(PERT_RANK_FILE, index_col=0)


def load_cell_ranks():
    return pd.read_csv(CELL_RANK_FILE, index_col=0)
