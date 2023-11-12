from models.main_models import *
from collections import defaultdict

#train and test on all 15 splits for the specified model, split, task, and feature type
def train_test_retrospective_model(model_type, split_type, task_type, feature_type):
    datasets_dir = f'{main_datasets_dir}/retrospective/splits'
    num_splits = 15
    res_list = []

    #loop over splits
    for i in range(num_splits):
        # get data and params, initialize model
        model_params = get_params(model_type, split_type, task_type, feature_type)
        test_df = pd.read_csv(f'{datasets_dir}/{split_type}/test_{i+1}.csv',
                              converters=converters)
       
        if model_type == 'nn':
            train_df = pd.read_csv(f'{datasets_dir}/{split_type}/train_{i+1}.csv',
                                   converters=converters)
            val_df = pd.read_csv(f'{datasets_dir}/{split_type}/val_{i+1}.csv',
                                 converters=converters)
            model = Model_Trainer(model_type, split_type, task_type, feature_type, model_params, train_df, val_df, test_df)
        else:
            trainval_df = pd.read_csv(f'{datasets_dir}/{split_type}/trainval_{i+1}.csv',
                                      converters=converters)
            model = Model_Trainer(model_type, split_type, task_type, feature_type, model_params, trainval_df, val_df=None, test_df=test_df)
        model.train_test_model()

        #save results
        res_list.append(model.res)
   
    #handle aggregated and per-split results
    res_agg, res_trials = defaultdict(list), defaultdict(list)
    for res in res_list:
        for key,value in res.items():
            res_trials[key].append(value)

    for key,value in res_trials.items():
        avg, std = np.mean(value), np.std(value)
        res_agg[key].append(avg)
        res_agg[key].append(std)
    return dict(res_agg), dict(res_trials)