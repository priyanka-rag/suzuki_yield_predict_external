from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
from functools import partial

from models.main_models import *

def hyperopt_output_to_params(best,hyper_dict):
    for key in best.keys():
        if key in hyper_dict.keys():
            if isinstance(hyper_dict[key],list):
                best[key] = hyper_dict[key][best[key]]             
    return best

def get_params(model_type):
    if model_type == 'rf':
        hyper_dict = {
            "n_estimators": (20,500)
        }

        parameter_space = {
            "n_estimators": scope.int(hp.quniform("n_estimators", hyper_dict['n_estimators'][0], hyper_dict['n_estimators'][1], 1))
        }

    elif model_type == 'xgb':
        hyper_dict = {
            "n_estimators": (20,500),
            "learning_rate": (1e-4,1e-1)
        }

        parameter_space = {
            "n_estimators": scope.int(hp.quniform("n_estimators", hyper_dict['n_estimators'][0], hyper_dict['n_estimators'][1], 1)),
            "learning_rate": hp.loguniform("learning_rate", np.log(hyper_dict['learning_rate'][0]), np.log(hyper_dict['learning_rate'][1]))
        }

    elif model_type == 'nn':
        hyper_dict = {
            "hidden_layer_sizes": [[1024,512,128], [512,256,64], [256, 128, 64], [512, 512, 256], [256, 256, 256]],
            "learning_rate_init": (1e-5,1e-2)
        }

        parameter_space =  {
            "hidden_layer_sizes": hp.choice("hidden_layer_sizes", hyper_dict['hidden_layer_sizes']),
            "learning_rate_init": hp.loguniform("learning_rate_init", np.log(hyper_dict['learning_rate_init'][0]), np.log(hyper_dict['learning_rate_init'][1]))
        }
        
    return hyper_dict,parameter_space

def model_eval(params, model_type, split_type, task_type, feature_type):
    #get data
    trainval_df = pd.read_csv(f'{main_datasets_dir}/retrospective/splits/{split_type}/trainval_1.csv',
                              converters=converters) #optimize hyperparameters for first split
    
    #train model over 5 train/val splits
    perf_metrics = []
    split_seeds = [0,15,30,45,60] #some random seeds for the train/val splitting
    for split_seed in split_seeds:
        train_df, val_df = split_data(trainval_df, split_type, 0.15/0.85, split_seed)
        if model_type == 'nn':
            train_df.reset_index()
            val_df.reset_index()
            train_df, valval_df = split_data(train_df, split_type, 0.15, 0)
            model = Model_Trainer(model_type, split_type, task_type, feature_type, params, train_df, val_df=valval_df, test_df=val_df)
        else: 
            model = Model_Trainer(model_type, split_type, task_type, feature_type, params, train_df, val_df=None, test_df=val_df)
        model.train_test_model()
        if task_type == 'bin' or task_type == 'mul': perf_metrics.append(model.res['Accuracy'])
        elif task_type == 'reg': perf_metrics.append(model.res['R2'])

    #get average metric across splits
    avg_metric = np.mean(perf_metrics)
    return -avg_metric

def perform_hyperopt_on_task(model_type, split_type, task_type, feature_type):
    hyper_dict, parameter_space = get_params(model_type)
    trials = Trials()
    hp_random_state = np.random.default_rng(0)
    fmin_objective = partial(model_eval, model_type=model_type, split_type=split_type, task_type=task_type, feature_type=feature_type)
    best = fmin(fmin_objective, parameter_space, algo=tpe.suggest, max_evals=25, trials=trials, rstate=hp_random_state)
    best_params = hyperopt_output_to_params(best, hyper_dict)
    best_score = -trials.best_trial['result']['loss']
    return best_params, best_score