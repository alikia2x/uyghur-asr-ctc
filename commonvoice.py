import pandas as pd
from umsc import UgMultiScriptConverter
from tqdm import tqdm

source_script = 'UAS'
target_script = 'ULS'
converter = UgMultiScriptConverter(source_script, target_script)

df = pd.read_csv('./data/commonvoice/validated.tsv', sep='\t')

new_df = pd.DataFrame(columns=['path', 'script'])

qbar = tqdm(total=len(df))
for index, row in df.iterrows():
    qbar.update(1)
    # Skip too bad samples
    if row['up_votes'] < row['down_votes']:
        continue
    new_df.loc[index] = [row['path'], converter(row['sentence'])]

new_df.to_csv('./data/training/commonvoice_train.csv', index=False)
