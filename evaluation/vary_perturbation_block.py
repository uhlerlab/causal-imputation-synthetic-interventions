from evaluation.helpers import PredictionManager, EvaluationManager
from src.algorithms2 import impute_unit_mean, impute_intervention_mean, impute_two_way_mean
from src.algorithms2 import predict_intervention_fixed_effect, predict_synthetic_control_unit_ols, predict_synthetic_control_unit_hsvt_ols
from src.algorithms2 import HSVTRegressor, synthetic_control_unit_inner, hsvt, approximate_rank
from line_profiler import LineProfiler
lp = LineProfiler()

algs = []

# pm = PredictionManager(0, 0, 0, 0, name='old_data')
# em = EvaluationManager(pm)
# pm.predict(impute_unit_mean)
# pm.predict(impute_intervention_mean)
# pm.predict(impute_two_way_mean)
# em.boxplot()
# em.boxplot_per_intervention()


for pert_start in [0]:
    pm = PredictionManager(
        num_cells=10,
        num_perts=200,
        name='level2_filtered_common_log2_minmax',
        num_folds=None
    )
    pm.predict(impute_unit_mean)
    pm.predict(impute_intervention_mean)
    pm.predict(impute_two_way_mean)
    pm.predict(predict_intervention_fixed_effect, control_intervention='DMSO')
    pm.predict(predict_synthetic_control_unit_ols, num_desired_interventions=None, progress=True)
    pm.predict(predict_synthetic_control_unit_hsvt_ols, num_desired_interventions=None, progress=True, overwrite=True)

    em = EvaluationManager(pm)
    # r = em.r2()
    # r = em.r2_in_iv()
    # em.r2_in_iv()
    em.boxplot()
    em.boxplot_per_intervention()


