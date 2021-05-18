from evaluation.helpers import PredictionManager, EvaluationManager
from src.algorithms import impute_unit_mean, impute_intervention_mean, impute_two_way_mean
from src.algorithms import predict_intervention_fixed_effect, predict_synthetic_intervention_ols, predict_synthetic_intervention_hsvt_ols
from src.algorithms.synthetic_interventions2 import predict_synthetic_intervention_ols
from src.algorithms import HSVTRegressor, synthetic_intervention_inner, hsvt, approximate_rank, fill_missing_means
from line_profiler import LineProfiler
from src.algorithms import impute_mice, impute_missforest
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
    pm = PredictionManager(
        num_cells=None,
        num_perts=num_perts,
        name=name,
        num_folds=None,
        average=average
    )
    # pm.predict(impute_missforest, overwrite=False)
    # pm.predict(impute_mice, overwrite=False)
    pm.predict(impute_unit_mean, overwrite=False)
    pm.predict(impute_intervention_mean, overwrite=False)
    pm.predict(impute_two_way_mean, overwrite=False)
    pm.predict(predict_intervention_fixed_effect, control_intervention='DMSO', overwrite=False)
    pm.predict(
        predict_synthetic_intervention_ols,
        num_desired_donors=None,
        donor_dim='intervention',
        progress=False,
        overwrite=True,
    )
    pm.predict(
        predict_synthetic_intervention_ols,
        num_desired_donors=None,
        donor_dim='unit',
        progress=False,
        overwrite=True,
    )
    # pm.predict(
    #     predict_synthetic_intervention_hsvt_ols,
    #     num_desired_donors=None,
    #     energy=.95,
    #     hypo_test=True,
    #     hypo_test_percent=.1,
    #     donor_dim='intervention',
    #     progress=False,
    #     overwrite=False,
    #     multithread=False,
    #     equal_rank=True
    # )
    if not average:
        energy = .95
        pm.predict(
            predict_synthetic_intervention_hsvt_ols,
            num_desired_donors=None,
            energy=energy,
            hypo_test=True,
            hypo_test_percent=.1,
            donor_dim='intervention',
            progress=False,
            overwrite=False,
            multithread=False,
            equal_rank=True
        )
        pm.predict(
            predict_synthetic_intervention_hsvt_ols,
            num_desired_donors=None,
            energy=energy,
            hypo_test=False,
            donor_dim='intervention',
            progress=False,
            overwrite=False,
            multithread=False
        )

    em = EvaluationManager(pm)
    r = em.r2()
    em.boxplot_relative_mse()
    # print(set(r.index.get_level_values('alg')))
    # r = em.r2_in_iv()
    # em.r2_in_iv()
    em.boxplot()
    em.boxplot_rmse()
    em.boxplot_per_intervention()
    em.plot_times()
    em.plot_quantile_relative_mse()
    # em.statistic_vs_best()

    return r


if __name__ == '__main__':
    # lp.add_function(predict_synthetic_control_unit_ols)
    # lp.add_function(synthetic_control_unit_inner)
    # lp.runcall(run, 'level2_common')
    # lp.print_stats()
    # run('level2_filtered_common')
    import itertools as itr

    r2 = run('level2', average=True, num_perts=100)
    # r2_ = run('level2', average=False, num_perts=100)

    # for average, num_perts in itr.product([True, False], [100]):
    #     run('level2', average=average, num_perts=num_perts)
    #     run('level2_filtered', average=average, num_perts=num_perts)
    #     run('level2_filtered_log2_minmax', average=average, num_perts=num_perts)
    # run('level2_filtered_distinct')
