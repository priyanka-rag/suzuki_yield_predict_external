from models.main_models import *

#remove libraries from a df (used for library design to remove the 2 libraries we present in the paper from the training data)
def remove_desired_libraries(df, libs_to_remove):
    idx_to_keep = []
    cores_to_remove = [df[df['Library'] == lib]['Core_Smiles'].to_list()[0] for lib in libs_to_remove]
    for i,row in df.iterrows():
        if row['Core_Smiles'] not in cores_to_remove and row['Monomer_Smiles'] not in cores_to_remove:
            idx_to_keep.append(i)
    df = df.iloc[idx_to_keep]
    return df

#generate predictions used in the monomer replacement experiments
def generate_library_predictions(mode):
    #initialize some values specific to the library design & rescue experiments
    model_type = 'rf'
    feature_type = 'fgpdft'
    if mode == 'design': split_type = 'core' #set split types just for the purpose of hyperparameters
    else: split_type = 'random'

    #read data
    retrospective_df = read_retrospective_data()
    monomer_replacement_df = read_monomer_replacement_data()
    test_df_copy = monomer_replacement_df #dummy copy of test dataset to store all predictions
    test_df_copy = test_df_copy.loc[:,:'Boronic_Acid']
    if mode == 'design': #if doing library design, remove some libraries from the training set
        libs_to_remove = pd.read_csv(f'{main_datasets_dir}/monomer_replacement/libraries_for_replacement.csv')['Library'].to_list()
        retrospective_df = remove_desired_libraries(retrospective_df, libs_to_remove)

    for task_type in ['bin', 'mul', 'reg']:
        print(task_type)
        # initialize model, train, and predict
        model_params = get_params(model_type, split_type, task_type, feature_type)
        if model_type == 'nn':
            train_df, val_df = split_data(retrospective_df, split_type, 0.15, 0)
            model = Model_Trainer(model_type, split_type, task_type, feature_type, model_params, train_df, val_df, test_df=monomer_replacement_df)
        else:
            model = Model_Trainer(model_type, split_type, task_type, feature_type, model_params, train_df=retrospective_df, val_df=None, test_df=monomer_replacement_df)
        model.train_test_model()
        if task_type == 'reg': model.y_pred = model.y_pred*100 #unscale regression targets
        test_df_copy[f'{task_type}_Predicted_Yield'] = list(model.y_pred)
    return test_df_copy