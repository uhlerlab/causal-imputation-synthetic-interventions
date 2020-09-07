import os
import pandas as pd

ROOT_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'raw')
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, 'processed')

LINCS2_EPSILON_FILE = os.path.join(RAW_DATA_FOLDER, 'GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx')
LINCS2_EPSILON_IMPUTED_FILE = os.path.join(PROCESSED_DATA_FOLDER, 'imputed_lincs2_epsilon.gctx')
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
    return pd.read_csv(INST_INFO_EPSILON_FILE, sep='\t')
