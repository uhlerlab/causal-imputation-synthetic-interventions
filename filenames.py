import os
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse

ROOT_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'raw')
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, 'processed')

LINCS2_EPSILON_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx')
LINCS2_EPSILON_IMPUTED_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978_imputed.gctx')
LINCS2_EPSILON_825_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978_825genes.gctx')

LINCS3_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx')
LINCS3_PRUNED_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328_pruned.gctx')

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
    return parse(LINCS2_EPSILON_IMPUTED_FILE).data_df


def load_cmap_filtered():
    return parse(LINCS2_EPSILON_825_FILE).data_df


def load_cmap_original():
    return parse(LINCS2_EPSILON_FILE).data_df


def load_cmap_level3_original():
    return parse(LINCS3_FILE).data_df


def load_cmap_level3():
    return parse(LINCS3_PRUNED_FILE).data_df
