import numpy as np
import os
from cmapPy.pandasGEXpress.parse import parse
from cmapPy.pandasGEXpress.write_gctx import write
from cmapPy.pandasGEXpress.GCToo import GCToo
import pandas as pd
from typing import Dict, Optional
from sklearn.preprocessing import minmax_scale
from filenames import *


def pandas_minmax(df, axis):
    return pd.DataFrame(minmax_scale(df, axis=axis), index=df.index, columns=df.columns)


class AverageManager:
    def __init__(
            self,
            folder: str,
            original_expressions: pd.DataFrame,
            spaces2encoders: Optional[Dict] = None,
            log2=True,
            minmax=True
    ):
        """
        Helper class for finding average gene expression vectors, after possibly:
        * encoding,
        * log2(x)+1 transformation
        * minmax scaling
        """
        self.folder = os.path.join('processing', 'helper_data', f'folder_log2={log2}_minmax={minmax}')
        self.original_expressions = original_expressions
        self.spaces2encoders = spaces2encoders if spaces2encoders is not None else {'original': lambda x: x}
        self.log2 = log2
        self.minmax = minmax

        os.makedirs(os.path.join(self.folder, 'averages'), exist_ok=True)

    def get_processed(self, overwrite=False):
        if not self.log2 and not self.minmax:
            return self.original_expressions

        processed_expressions_filename = os.path.join(self.folder, 'processed.gctx')
        if overwrite or not os.path.exists(processed_expressions_filename):
            processed_expressions = self.original_expressions
            if self.log2:
                print('[AverageManager.get_processed] log2')
                processed_expressions = np.log2(processed_expressions + 1)
            elif self.minmax:
                print('[AverageManager.get_processed] minmax')
                processed_expressions = pandas_minmax(processed_expressions, axis=0)
        else:
            c = parse(processed_expressions_filename)
            processed_expressions = c.data_df
        print(f'[AverageManager.get_processed] processed shape: {processed_expressions.shape}')

        return processed_expressions

    def get_encoded(self, space):
        if space == 'original':
            return self.get_processed()

        encoded_filename = os.path.join(self.folder, f'{space}.gctx')
        if os.path.exists(encoded_filename):
            c = parse(encoded_filename)
            gene_expression_df = c.data_df
        else:
            encoder = self.spaces2encoders[space]
            print(f'[AverageManager.get_encoded] encoding!')
            gene_expression_df = encoder(self.get_processed())
            print(f'[AverageManager.get_encoded] writing to {encoded_filename}')
            gctoo = GCToo(gene_expression_df)
            write(gctoo, encoded_filename)
        print(f'[AverageManager.get_encoded] encoded shape: {gene_expression_df.shape}')

        return gene_expression_df

    def get_averages_filename(self, space):
        return os.path.join(self.folder, 'averages', f'averages_{space}.pkl')

    def get_averages(self, space, overwrite=False):
        averages_filename = self.get_averages_filename(space=space)

        if overwrite or not os.path.exists(averages_filename):
            print(f'====== COMPUTING AVERAGES FOR {space} ======')
            gene_expression_df = self.get_encoded(space)
            inst_info = pd.read_csv(INST_INFO_FILE, sep='\t', index_col=0)

            print(f'[AverageManager.get_averages] Grouping inst_id by cell_id and {PERT_ID_FIELD}')
            inst_info_no_meta = inst_info.set_index(['cell_id', PERT_ID_FIELD])
            ids2ixs_no_meta = dict(zip(inst_info.index, inst_info_no_meta.index))

            print('[AverageManager.get_averages] Averaging')
            average_df = gene_expression_df.groupby(ids2ixs_no_meta, axis=1).mean()

            print(f'[AverageManager.get_averages] Means shape: {average_df.shape}')
            average_df = pd.DataFrame(
                average_df.values.T,
                columns=average_df.index,
                index=pd.MultiIndex.from_tuples(average_df.columns)
            )
            average_df.index.set_names(['unit', 'intervention'], inplace=True)

            print(f'[AverageManager.get_averages] Removing units without control')
            control_df = average_df[average_df.index.get_level_values('intervention') == 'DMSO']
            units_with_control = set(control_df.index.get_level_values('unit'))
            average_df = average_df[average_df.index.get_level_values('unit').isin(units_with_control)]

            print(f'[AverageManager.get_averages] Saving averages, shape: {average_df.shape}')
            average_df.to_pickle(averages_filename)
        else:
            average_df = pd.read_pickle(averages_filename)

        return average_df

    def get_space2average_df(self, overwrite=False):
        return {space: self.get_averages(space, overwrite=overwrite) for space in self.spaces2encoders.keys()}
