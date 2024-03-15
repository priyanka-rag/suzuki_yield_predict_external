from models.main_models import *

def split_post_2021_data(retrospective_df, post_2021_df):
    samec_samem_ix, samec_diffm_ix, diffc_samem_ix, diffc_diffm_ix = [], [], [], []
    original_cores = list(set(retrospective_df['Core_Smiles'].to_list()))
    original_monomers = list(set(retrospective_df['Monomer_Smiles'].to_list()))
    retrospective_molecules = list(set(original_cores + original_monomers))
    for i,row in post_2021_df.iterrows():
        core, monomer = row['Core_Smiles'], row['Monomer_Smiles']
        if (core in retrospective_molecules and monomer in retrospective_molecules): 
            samec_samem_ix.append(i)
        elif (core in retrospective_molecules and monomer not in retrospective_molecules): 
            samec_diffm_ix.append(i)
        elif (core not in retrospective_molecules and monomer in retrospective_molecules): 
            diffc_samem_ix.append(i)
        elif (core not in retrospective_molecules and monomer not in retrospective_molecules): 
            diffc_diffm_ix.append(i)
    return [samec_samem_ix, samec_diffm_ix, diffc_samem_ix, diffc_diffm_ix]

def get_naive_score(yields, task_type):
    if task_type == 'bin':
        score = sum(yields)/len(yields)
    elif task_type == 'mul':
        score = sum([1 if y == 0 else 0 for y in yields])/len(yields)
    return score

#train on full retrospective dataset and test on post-mid-2021 dataset
def test_on_post_2021_data(model_type, task_type, feature_type, split_preds):
    #read data
    retrospective_df = read_retrospective_data()
    post_2021_df = read_post_2021_data()
    split_type = 'core' # this represents an extrapolative task, so set the split_type as core
    model_params = get_params(model_type, split_type, task_type, feature_type)

    # initialize model, train, and predict
    if model_type == 'nn':
        train_df, val_df = split_data(retrospective_df, split_type, 0.15, 0)
        model = Model_Trainer(model_type, split_type, task_type, feature_type, model_params, train_df, val_df, test_df=post_2021_df)
    else: 
        model = Model_Trainer(model_type, split_type, task_type, feature_type, model_params, train_df=retrospective_df, val_df=None, test_df=post_2021_df)
    model.train_test_model()
    res = model.res

    if split_preds:
        split_res = []
        split_ixs = split_post_2021_data(retrospective_df, post_2021_df)
        for ix in split_ixs:
            y_test_sub, y_pred_sub, y_probs_sub = model.y_test[ix], model.y_pred[ix], model.y_probs[ix]
            if task_type != 'reg': naive_score = get_naive_score(y_test_sub, task_type)
            split_res.append(score_predictions(task_type, y_test_sub, y_pred_sub, y_probs_sub))
        return res, split_res
    else:
        return res