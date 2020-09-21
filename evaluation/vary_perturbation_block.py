from evaluation.helpers import PredictionManager, EvaluationManager
from src.algorithms2 import impute_unit_mean, impute_intervention_mean, impute_two_way_mean, predict_intervention_fixed_effect, predict_synthetic_control_unit

algs = []
pert_start = 0
pert_stop = 20
pm = PredictionManager(0, 10, pert_start, pert_stop)
pm.predict(impute_unit_mean)
pm.predict(impute_intervention_mean)
pm.predict(impute_two_way_mean)
pm.predict(predict_intervention_fixed_effect, control_intervention='DMSO')
pm.predict(predict_synthetic_control_unit, num_desired_interventions=3, progress=True)
