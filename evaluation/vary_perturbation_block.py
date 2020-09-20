from evaluation.helpers import PredictionManager, EvaluationManager
from src.algorithms2 import impute_unit_mean, impute_intervention_mean, impute_two_way_mean

algs = []
pert_start = 0
pert_stop = 10
pm = PredictionManager(0, 10, pert_start, pert_stop)
pm.predict(impute_unit_mean)
pm.predict(impute_intervention_mean)
pm.predict(impute_two_way_mean)

