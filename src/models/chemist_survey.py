from models.main_models import *
import pdb

#generate model predictions used in the chemist survey
def gen_survey_predictions(split_type):
    model_type = 'rf'
    feature_type = 'fgpdft'
    test_frac = 0.3
    split_seed = 0 
    df = read_retrospective_data()
    train_df, test_df = split_data(df, split_type, test_frac, split_seed)
    test_df_copy = test_df #dummy copy to record predictions
    test_df_copy = test_df_copy.loc[:,:'Boronic_Acid']
    for task_type in ['bin', 'mul', 'reg']:
        print(task_type)
        # initialize model, train, and predict
        model_params = get_params(model_type, split_type, task_type, feature_type)
        model = Model_Trainer(model_type, split_type, task_type, feature_type, model_params, train_df, val_df=None, test_df=test_df)
        model.train_test_model()
        if task_type == 'reg': model.y_pred = model.y_pred*100 #unscale regression targets
        test_df_copy[f'{task_type}_Predicted_Yield'] = list(model.y_pred)
    return test_df_copy
