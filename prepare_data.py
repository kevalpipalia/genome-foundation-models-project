import pandas as pd
from Bio import SeqIO
import numpy as np
import gzip

normalized_expression_values_file = 'data/Whole_Blood.v8.normalized_expression.bed.gz'

with gzip.open(normalized_expression_values_file) as file:
    # read file into a pandas DataFrame
    # BED format is typically tab-delimited; no header row
    df = pd.read_csv(file, sep='\t', index_col=0)


# we will show predictions in first 100 TSS sites for predicting expressions, Since we are finetuning separate model for each site.
for tss_seq in range(0, 100):
    df_tss_seq = df.iloc[[tss_seq], :]
    tss_start = df_tss_seq.iloc[0, 1]
    df_tss_seq = df_tss_seq.drop(['start', 'end', 'gene_id'], axis = 1)
    df_tss_seq = df_tss_seq.T
    df_tss_seq['sequence'] = ''

    for chr, val in df_tss_seq.iterrows():
        file_path='data/newversion/{}_sequences_windowed.fasta'.format(chr.split('-')[1])
        seq = SeqIO.parse(file_path, "fasta")
        first = next(seq)
        sequence = first.seq
        sequence = str(sequence)
        sequence = sequence.upper()
        df_tss_seq.loc[chr, 'sequence'] = sequence

    print(tss_start)
    df_tss_seq.to_csv('data/processed_data/{}.csv'.format(tss_start), index=False)