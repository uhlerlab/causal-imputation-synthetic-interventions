"""
Run the diagnostic on the most common interventions
"""
from processing.most_common_manager import MostCommonManager
from processing.average_manager import AverageManager
from src.predict import diagnostic
import pandas as pd

nperts = 10
mc_manager = MostCommonManager(None, nperts)
avg_manager = AverageManager(
    f'{nperts}_most_common_perts',
    mc_manager.get_most_common_gctx(),
    log2=False,
    minmax=False
)
avgs: pd.DataFrame = avg_manager.get_space2average_df(overwrite=True)['original']
avgs = avgs.reset_index(['intervention', 'unit'])

pre_df = avgs[avgs['intervention'] == 'DMSO']
post_df = avgs[avgs['intervention'] != 'DMSO']
pre_df['metric'] = 'm0'
post_df['metric'] = 'm0'

diag_results = diagnostic(
    pre_df,
    post_df
)
print(f'Number of passing results: {(diag_results["pvalues_test"] == "Pass").sum()}')
