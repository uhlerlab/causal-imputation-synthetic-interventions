from filenames import load_cmap_original

data = load_cmap_original()
num_dropouts_df = (data == 1).sum(axis=1).sort_values(ascending=False)
num_dropouts_df = num_dropouts_df[num_dropouts_df != 0]
num_dropouts_df.to_csv('scratch/num_dropouts.csv')
