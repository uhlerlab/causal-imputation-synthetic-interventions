from processing.most_common_manager import MostCommonManager
from processing.average_manager import AverageManager
from evaluation.prediction_manager import PredictionManager
from src.predict import fill_tensor

mc_manager = MostCommonManager(10, 20)
avg_manager = AverageManager('test', mc_manager.get_most_common_gctx(), log2=False, minmax=False)
avgs = avg_manager.get_space2average_df()['original']

p_manager = PredictionManager('test', avgs, 'DMSO')
preds = p_manager.predict(fill_tensor, 'hsvt_ols')
