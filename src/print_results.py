import pandas as pd
import os

results_dir = '../models/unfrozen_bert_linear_lmbd_0.5'

runs = os.listdir(results_dir)

for run in runs:
    results_path = os.path.join(results_dir, run, 'results.tsv')

    results_df = pd.read_csv(results_path, sep='\t')

    print(run)
    print(results_df.mean())
    print('\n')
