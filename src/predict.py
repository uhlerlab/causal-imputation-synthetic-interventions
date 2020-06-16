import numpy as np
import pandas as pd
from src.diagnostic import diagnostic_test
from src.algorithms import hsvt_ols
from tqdm import tqdm


# DIAGNOSTIC TESTS
def diagnostic(pre_df, post_df, t=0.99, alpha=0.05):
	columns = ['unit', 'intervention', 'metric']
	
	# sort dataframes 
	pre_df = pre_df.sort_values(by=columns)
	post_df = post_df.sort_values(by=columns)

	# get ivs, units, metrics
	ivs = np.sort(pd.unique(post_df.intervention))
	units = list(np.sort(pd.unique(pre_df.unit)))
	metrics = list(np.sort(pd.unique(pre_df.metric)))

	# get number of units and interventions
	N, K, M = len(units), len(ivs), len(metrics)

	# initialize 
	diagnostic_rslts = np.empty((K*M, 2))
	diagnostic_rslts[:] = np.nan

	# perform diagnostic tests
	for i, iv in enumerate(ivs):
		unit_ids = pd.unique(post_df[post_df.intervention==iv]['unit'])

		for m, metric in enumerate(metrics): 
			diagnostic_rslts[i*M+m, :] = diagnostic_test(pre_df, post_df, unit_ids, metric, iv, t=t, alpha=alpha) 

	# create output dataframe
	df = pd.DataFrame(data=diagnostic_rslts, columns=['pvalues_test', 'energy_statistic'])
	diag_ivs = [ivs[k // M] for k in range(K*M)]
	diag_metrics = metrics * K
	df.insert(0, 'metric', diag_metrics)
	df.insert(0, 'intervention', diag_ivs)
	df['pvalues_test'] = df['pvalues_test'].replace(0, "Fail")
	df['pvalues_test'] = df['pvalues_test'].replace(1, "Pass")
	df['pvalues_test'] = df['pvalues_test'].replace(np.nan, "N/A")
	return df 


# PREDICT COUNTERFACTUALS
# Modified from Abdullah's version of fill_tensor.
def fill_tensor(pre_df, post_df, t=0.99, center=True, rcond=1e-15, alpha=0.05, include_pre=True, \
                rank_method = 'spectral_energy', return_donors_info = False, targets=None):
	donor_column = ('donor' in pre_df.columns) & ('donor' in post_df.columns)
	columns = ['unit', 'intervention', 'metric']
	if donor_column: 
		columns = ['unit', 'intervention', 'metric', 'donor']
	metric='m0'    
	
	# sort dataframes by (unit, intervention)
	pre_df = pre_df.sort_values(by=columns)
	post_df = post_df.sort_values(by=columns)

	# get all unique interventions (from post-intervention dataframe)
	ivs = np.sort(pd.unique(post_df.intervention))

	# get all units (using pre-intervention data)
	units = list(np.sort(pd.unique(pre_df.unit)))

	# get all metrics
	metrics = list(np.sort(pd.unique(pre_df.metric)))
    
	pairs = dict()
	if targets == None:
		for iv in ivs:
			pairs[iv] = units
	else:
		for unit, iv in targets:
			if iv not in pairs.keys():
				pairs[iv] = []
			pairs[iv].append(unit)

	# get number of units and interventions
	N, K, M = len(units), len(ivs), len(metrics)
	T0 = pre_df.shape[1]-len(columns)
	T = T0 + post_df.shape[1]-len(columns)

	# check to make sure there aren't any duplicate units in the pre-intervention dataframe
	assert len(pre_df.unit.unique()) == N

    # Cindy: create dictionary of all interventions received by unit:
	ivs_dict = {}
	for unit in units:
		unit_ivs = pd.unique(post_df[post_df.unit==unit]['intervention'])
		ivs_dict[unit] = set(unit_ivs)

	# initialize output dataframe
	yh_data = np.array([])
	idx_data = np.array([])
	donors_dict = {}
	for iv in tqdm(pairs.keys()):
		donors_dict[iv] ={}
		# get potential donors (units who receive intervention 'iv') from POST-intervention data
		unit_ids = pd.unique(post_df[post_df.intervention==iv]['unit'])
		if donor_column:
				donor_list = pd.unique(post_df[post_df.donor==1]['unit'])

		for unit in pairs[iv]: 
			donors_dict[iv][unit] = {}
			# exclude (target) unit from being included in (donor) unit_ids
			donor_units = unit_ids[unit_ids != unit] if unit in unit_ids else unit_ids
			if donor_column:
				 donor_units = donor_units[np.isin(donor_units,donor_list)]
			num_donors = len(donor_units)
            
			donor_ivs = set.intersection(*([ivs_dict[unit] for unit in unit_ids] + [unit]))

            ## get donor data
            # get pre-intervention data
			X1_pre = pre_df.loc[(pre_df.unit.isin(donor_units))]
			donors1 = X1_pre.unit.values
            # get post-intervention data from donor therapies
			X1_post = post_df.loc[(post_df.unit.isin(donor_units)) & (post_df.intervention.isin(donor_ivs)) & ~(post_df.intervention==iv)]
			X1 = pd.concat([X1_pre, X1_post], axis=0)
			X1 = X1.drop(columns=['metric']).set_index(['unit','intervention']).unstack('intervention')
			X1.columns = X1.columns.swaplevel(0,1)
			X1.sort_index(axis=1, level=0, inplace=True)
			X1 = X1.values           

            ## get target data
            # get pre-intervention data
			X2_pre = pre_df.loc[(pre_df.unit==unit)]
			# get post-intervention data from donor therapies
			X2_post = post_df.loc[(post_df.unit==unit)&(post_df.intervention.isin(donor_ivs))& ~(post_df.intervention==iv)]
			X2 = pd.concat([X2_pre, X2_post],axis=0)
			X2 = X2.drop(columns=['metric']).set_index(['unit','intervention']).unstack('intervention')
			X2.columns = X2.columns.swaplevel(0,1)
			X2.sort_index(axis=1, level=0, inplace=True)
			X2 = X2.values

			y1 = post_df.loc[(post_df.unit.isin(donors1)) & (post_df.intervention==iv)].drop(columns=columns).values
    
			yh = hsvt_ols(X1, X2, y1, t=t, center=center, rcond=rcond, include_pre=include_pre, method = rank_method, return_coefficients=False)
			# append data
			try:
				yh_data = np.vstack([yh_data, yh]) if yh_data.size else yh
			except:
				continue
			idx_data = np.vstack([idx_data, [unit, iv, metric]]) if idx_data.size else np.array([unit, iv, metric])
		    #print("unit:%s, metric:%s, intervention: %s, donors: %s"%(unit, metric, i ,donor_units))
            
            
	post_cols = list(post_df.drop(columns=columns).columns)
	df_columns = post_cols
	df_synth = pd.DataFrame(columns=df_columns, data=yh_data)

	if len(targets)==1:
		df_synth.insert(0, 'metric', idx_data[2])
		df_synth.insert(0, 'intervention', idx_data[1])
		df_synth.insert(0, 'unit', idx_data[0])
	else:
		df_synth.insert(0, 'metric', idx_data[:, 2])
		df_synth.insert(0, 'intervention', idx_data[:, 1])
		df_synth.insert(0, 'unit', idx_data[:, 0])
	if return_donors_info: return df_synth, donors_dict
	return df_synth





