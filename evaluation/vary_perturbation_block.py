from evaluation.helpers import PredictionManager, EvaluationManager
from src.algorithms2 import impute_unit_mean, impute_intervention_mean, impute_two_way_mean
from src.algorithms2 import predict_intervention_fixed_effect, predict_synthetic_control_unit_ols, predict_synthetic_control_unit_hsvt_ols
from src.algorithms2 import HSVTRegressor, synthetic_control_unit_inner, hsvt, approximate_rank, fill_missing_means
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


def run(name):
    for pert_start in [0]:
        pm = PredictionManager(
            num_cells=None,
            num_perts=100,
            name=name,
            num_folds=None
        )
        pm.predict(impute_unit_mean, overwrite=False)
        pm.predict(impute_intervention_mean, overwrite=False)
        pm.predict(impute_two_way_mean)
        pm.predict(predict_intervention_fixed_effect, control_intervention='DMSO', overwrite=False)
        pm.predict(predict_synthetic_control_unit_ols, num_desired_interventions=None, progress=False, overwrite=False)
        pm.predict(predict_synthetic_control_unit_hsvt_ols, num_desired_interventions=None, progress=False, overwrite=True, energy=.999, multithread=False)
        pm.predict(predict_synthetic_control_unit_hsvt_ols, num_desired_interventions=None, progress=False, overwrite=True, energy=.99, multithread=False)
        pm.predict(predict_synthetic_control_unit_hsvt_ols, num_desired_interventions=None, progress=False, overwrite=True, energy=.95, multithread=False)

        em = EvaluationManager(pm)
        # r = em.r2()
        # r = em.r2_in_iv()
        # em.r2_in_iv()
        em.boxplot()
        em.boxplot_per_intervention()


if __name__ == '__main__':
    # lp.add_function(predict_synthetic_control_unit_ols)
    # lp.add_function(synthetic_control_unit_inner)
    # lp.runcall(run, 'level2_common')
    # lp.print_stats()
    run('level2_filtered_common')
    # run('level2')
    # run('level2_filtered')
    # run('level2_filtered_log2_minmax')
    # run('level2_filtered_common_log2_minmax')
