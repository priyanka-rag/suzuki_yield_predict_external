import chemprop

from data.read_data import *
from data.featurize import *
from models.main_models import score_predictions

def unscale(task_type, y_pred):
    if task_type == 'bin':
        y_pred = np.array([0 if y < 0.5 else 1 for y in list(y_pred)])
    elif task_type == 'mul':
        y_pred = np.argmax(y_pred,axis=2)
    return y_pred

def expand_label_type(task_type):
    if task_type == 'bin': label_type_exp = 'classification'
    elif task_type == 'mul': label_type_exp = 'multiclass'
    elif task_type == 'reg': label_type_exp = 'regression'
    return label_type_exp

def train_test_chemprop(task_type,split_type,split_number):
    label_type_exp = expand_label_type(task_type)
    if task_type == 'bin': metric = 'accuracy'
    elif task_type == 'mul': metric = 'accuracy'
    elif task_type == 'reg': metric = 'r2'

    data_dir = f'{main_datasets_dir}/retrospective/splits/{split_type}'
    save_dir = f'{main_datasets_dir}/trained_gnn_models/{split_type}'

    #train
    train_arguments = [
    '--data_path', f'{data_dir}/train_{split_number}.csv',
    '--separate_val_path', f'{data_dir}/val_{split_number}.csv',
    '--separate_test_path', f'{data_dir}/test_{split_number}.csv',
    '--dataset_type', f'{label_type_exp}',
    '--smiles_columns', 'Atom_Mapped_Reaction',
    '--reaction',
    '--reaction_mode', 'reac_prod',
    '--target_columns', f'{task_type}_Percent_Yield',
    '--metric', metric,
    '--multiclass_num_classes', '4',
    '--save_dir', save_dir,
    '--batch_size', '512',
    '--pytorch_seed', '42',
    '--gpu', '1', 
    '--epochs', '100'
    ]

    args = chemprop.args.TrainArgs().parse_args(train_arguments)
    chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

    #predict
    test_arguments = [
        '--test_path', f'{data_dir}/test_{split_number}.csv',
        '--smiles_columns', 'Atom_Mapped_Reaction',
        '--batch_size', '512',
        '--checkpoint_dir', save_dir,
        '--preds_path', f'{save_dir}/test_preds.csv'
    ]

    test_df = pd.read_csv(f'{data_dir}/test_{split_number}.csv')
    y_test = np.array(test_df[f'{task_type}_Percent_Yield'].to_list())
    args = chemprop.args.PredictArgs().parse_args(test_arguments)
    y_pred = np.array(chemprop.train.make_predictions(args=args))
    y_pred = unscale(task_type, y_pred)
    res = score_predictions(task_type, y_test, y_pred)
    return res
