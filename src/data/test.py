from data.read_data import *
from data.featurize import *
from collections import Counter
import statistics
import pdb

df = read_retrospective_data()
lib = 'H6692-1'
df = pd.read_csv('/data/abbvie/dataset_files/retrospective/suzuki_data.csv')
df = df.loc[df['Library'] == lib]
yields = get_yield_data(df['Percent_Yield'].to_list(), 'bin')
yields.sort()
print(yields)
pdb.set_trace()
task_type = 'bin'
mon_df = pd.read_csv(f'{main_datasets_dir}/chemist_survey/rf_fgpdft_monomersplit_predictions.csv')
sub_df = df.loc[df['Library'] == lib]
syn_ids = mon_df.loc[mon_df['Library'] == lib]['Synthesis_ID'].to_list()
train_df = sub_df[~sub_df['Synthesis_ID'].isin(syn_ids)]
train_yields = get_yield_data(train_df['Percent_Yield'].to_list(), task_type)
print(get_yield_data(sub_df['Percent_Yield'].to_list(), task_type))
freq_class = statistics.mode(train_yields)
print(freq_class)

test_yields = get_yield_data(mon_df.loc[mon_df['Library'] == lib]['Percent_Yield'].to_list(), task_type)
print(len(test_yields))
test_yields.sort()
print(test_yields)