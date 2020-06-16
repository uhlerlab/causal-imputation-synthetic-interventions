from processing.most_common_manager import MostCommonManager
from processing.average_manager import AverageManager

mc_manager = MostCommonManager(10, 20)
avg_manager = AverageManager('test', mc_manager.get_most_common_gctx(), log2=False, minmax=False)
avgs = avg_manager.get_space2average_df()
