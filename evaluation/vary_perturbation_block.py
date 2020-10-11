from evaluation.helpers import PredictionManager, EvaluationManager
from src.algorithms import impute_unit_mean, impute_intervention_mean, impute_two_way_mean
from src.algorithms import predict_intervention_fixed_effect, predict_synthetic_intervention_ols, predict_synthetic_intervention_hsvt_ols
from src.algorithms import HSVTRegressor, synthetic_intervention_inner, hsvt, approximate_rank, fill_missing_means
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


def run(name, average, num_perts):
    for pert_start in [0]:
        pm = PredictionManager(
            num_cells=2,
            num_perts=num_perts,
            name=name,
            num_folds=None,
            average=average
        )
        pm.predict(impute_unit_mean, overwrite=False)
        pm.predict(impute_intervention_mean, overwrite=False)
        pm.predict(impute_two_way_mean)
        pm.predict(predict_intervention_fixed_effect, control_intervention='DMSO', overwrite=False)
        pm.predict(predict_synthetic_intervention_ols, num_desired_interventions=None, progress=False, overwrite=True, donor_dim='intervention')
        # pm.predict(predict_synthetic_intervention_ols, num_desired_interventions=None, progress=False, overwrite=True, donor_dim='unit')
        # energy = .95
        # pm.predict(
        #     predict_synthetic_intervention_hsvt_ols,
        #     num_desired_interventions=None,
        #     energy=energy,
        #     hypo_test=True,
        #     hypo_test_percent=.1,
        #     progress=False,
        #     overwrite=True,
        #     multithread=False
        # )
        # pm.predict(
        #     predict_synthetic_intervention_hsvt_ols,
        #     num_desired_interventions=None,
        #     energy=energy,
        #     hypo_test=False,
        #     progress=False,
        #     overwrite=False,
        #     multithread=False
        # )

        em = EvaluationManager(pm)
        r = em.r2()
        print(set(r.index.get_level_values('alg')))
        # r = em.r2_in_iv()
        # em.r2_in_iv()
        em.boxplot()
        em.boxplot_per_intervention()

        return r


if __name__ == '__main__':
    # lp.add_function(predict_synthetic_control_unit_ols)
    # lp.add_function(synthetic_control_unit_inner)
    # lp.runcall(run, 'level2_common')
    # lp.print_stats()
    # run('level2_filtered_common')
    import itertools as itr

    r2 = run('level2', average=True, num_perts=5)

    # for average, num_perts in itr.product([True, False], [100]):
    #     run('level2', average=average, num_perts=num_perts)
    #     run('level2_filtered', average=average, num_perts=num_perts)
    #     run('level2_filtered_log2_minmax', average=average, num_perts=num_perts)
    # run('level2_filtered_distinct')
