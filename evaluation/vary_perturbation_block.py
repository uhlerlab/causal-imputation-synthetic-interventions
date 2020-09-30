from evaluation.helpers import PredictionManager, EvaluationManager
from src.algorithms2 import impute_unit_mean, impute_intervention_mean, impute_two_way_mean, predict_intervention_fixed_effect, predict_synthetic_control_unit
from line_profiler import LineProfiler
lp = LineProfiler()

algs = []


for pert_start in [0]:
    pm = PredictionManager(0, 10, pert_start, pert_start+20, name='level2_common')
    pm.predict(impute_unit_mean)
    pm.predict(impute_intervention_mean)
    pm.predict(impute_two_way_mean)
    pm.predict(predict_intervention_fixed_effect, control_intervention='DMSO')
    pm.predict(predict_synthetic_control_unit, num_desired_interventions=1, progress=True)
    pm.predict(predict_synthetic_control_unit, num_desired_interventions=3, progress=True)
    pm.predict(predict_synthetic_control_unit, num_desired_interventions=None, progress=True)

    em = EvaluationManager(pm)
    # r = em.r2()
    # r = em.r2_in_iv()
    # em.r2_in_iv()
    em.boxplot()
    em.boxplot_per_intervention()


