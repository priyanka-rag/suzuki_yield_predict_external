import numpy as np
from sklearn.model_selection import train_test_split,GroupShuffleSplit

def split_data(df,split_type,test_frac,split_seed): 
    if split_type == 'random':
            train_ix, test_ix = train_test_split(np.arange(df.shape[0]),test_size=test_frac,random_state=split_seed)
    elif split_type == 'core':
        gs = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=split_seed)
        train_ix, test_ix = next(gs.split(df, y=None, groups=df.Core_Smiles)) #only 1 split, just grab it
    elif split_type == 'monomer':
        gs = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=split_seed)
        train_ix, test_ix = next(gs.split(df, y=None, groups=df.Monomer_Smiles)) #only 1 split, just grab it
    
    train_df = df.iloc[train_ix]
    test_df = df.iloc[test_ix]
    return train_df, test_df