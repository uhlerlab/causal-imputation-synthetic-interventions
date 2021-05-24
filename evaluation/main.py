from evaluation.helpers import PredictionManager, EvaluationManager
from src.algorithms import impute_unit_mean, impute_intervention_mean, impute_two_way_mean
from src.algorithms import predict_intervention_fixed_effect, predict_synthetic_intervention_ols
from src.algorithms.synthetic_interventions2 import predict_synthetic_intervention_ols
from visuals.plot_availability_matrix import plot_availability_matrix
from src.algorithms import impute_mice


def run(name, average, num_perts, num_cells=None):
    pm = PredictionManager(
        cell_start=0,
        num_cells=num_cells,
        num_perts=num_perts,
        name=name,
        num_folds=None,
        average=average
    )
    # plot_availability_matrix(pm.gene_expression_df, savefig=pm.result_string)
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
        overwrite=False,
    )
    pm.predict(
        predict_synthetic_intervention_ols,
        num_desired_donors=None,
        donor_dim='unit',
        progress=False,
        overwrite=False,
    )

    em = EvaluationManager(pm)
    em.plots("alg=predict_synthetic_intervention_ols,num_desired_donors=None,donor_dim=intervention")

    return pm, em


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="store", dest="name", default="level2", type=str)
    parser.add_argument("--average", action="store", default=True, type=bool)
    parser.add_argument("--num_perts", action="store", default=10, type=int)
    parser.add_argument("--num_cells", action="store", default=None, type=int)
    args = parser.parse_args()

    pm, em = run(args.name, args.average, args.num_perts, args.num_cells)
    df = em.relative_mse()
    alg = "alg=predict_synthetic_intervention_ols,num_desired_donors=None,donor_dim=intervention"
    df = df[df.index.get_level_values("alg") == alg]
    print(df.sort_values(0))


