import pandas as pd
import os

results_dir = '../models/unfrozen_bert_linearlmbd1.0_classic'

runs = os.listdir(results_dir)

for run in runs:

    results_path = os.path.join(results_dir, run, 'results.tsv')

    results_df = pd.read_csv(results_path, sep='\t')

    results_df = results_df[
        [
            'accuracy',
            'precision_scores_sarcastic_yes',
            'recall_scores_sarcastic_yes',
            'f1_scores_sarcastic_yes',
            'precision_scores_sarcastic_no',
            'recall_scores_sarcastic_no',
            'f1_scores_sarcastic_no',
        ]
    ]

    values = [round(x, 3) for x in results_df.mean().tolist()]

    print(run)
    print(values)
    print('\n')
