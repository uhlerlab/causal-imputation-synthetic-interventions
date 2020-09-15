from cmapPy.pandasGEXpress.parse import parse
from filenames import load_inst_info


def main(name, cmap_file):
    data = parse(cmap_file).data_df
    inst_info = load_inst_info()
    pass


if __name__ == '__main__':
    from filenames import LINCS2_EPSILON_FILE, LINCS2_EPSILON_IMPUTED_FILE, LINCS2_EPSILON_825_FILE, LINCS3_PRUNED_FILE

    files = {
        'original_level2': LINCS2_EPSILON_FILE,
        'imputed_level2': LINCS2_EPSILON_IMPUTED_FILE,
        'filtered_level2': LINCS2_EPSILON_825_FILE,
        'level3': LINCS3_PRUNED_FILE,
    }
