from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error,accuracy_score,precision_score,recall_score,roc_auc_score,average_precision_score,r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from data.read_data import *
from data.split import *
from data.featurize import *
from models.ffnn import *

def get_params(model_type, split_type, task_type, feature_type):
   df = pd.read_csv(f'{main_datasets_dir}/retrospective/optimal_hyperparameters.csv', converters={'Params': literal_eval})
   sub_df = df[(df['Model'] == model_type) & \
            (df['Split'] == split_type) & \
            (df['Task'] == task_type) & \
            (df['Feature'] == feature_type)]
   params = sub_df.iloc[0]['Params']
   return params

def score_predictions(task_type,y_test,y_pred):
   res = dict()
   if task_type == 'bin':
      res['Accuracy'] = accuracy_score(y_test, y_pred)
      res['Precision'] = precision_score(y_test, y_pred)
      res['Recall'] = recall_score(y_test, y_pred)
      res['AUROC'] = roc_auc_score(y_test, y_pred)
      res['AUPRC'] = average_precision_score(y_test, y_pred)
   elif task_type == 'mul':
      res['Accuracy'] = accuracy_score(y_test, y_pred)
   elif task_type == 'reg':
      res['R2'] = r2_score(y_test, y_pred)
      res['MAE'] = mean_absolute_error(y_test, y_pred)
      res['RMSE'] = mean_squared_error(y_test, y_pred, squared=False)
   return res

class Model_Trainer():
   def __init__(self, model_type, split_type, task_type, feature_type, model_params, train_df, val_df, test_df):
      self.model_type = model_type
      self.split_type = split_type
      self.task_type = task_type
      self.feature_type = feature_type
      self.train_df = train_df
      self.val_df = val_df
      self.test_df = test_df
      self.model_params = model_params
   
   def get_model_inputs(self):
      # read in everything
      yield_train = self.train_df['Percent_Yield']
      self.y_train = get_yield_data(yield_train, self.task_type)
      try: #for the prospective studies, we don't have ground truth yields for the test set
         yield_test = self.test_df['Percent_Yield']
      except:
         self.y_test = None
      else:
         self.y_test = get_yield_data(yield_test, self.task_type)

      boronic_acid_train, boronic_acid_test = self.train_df['Boronic_Acid'].to_list(), self.test_df['Boronic_Acid'].to_list()
      aryl_halide_train, aryl_halide_test = self.train_df['Aryl_Halide'].to_list(), self.test_df['Aryl_Halide'].to_list()
      
      catalyst_train,base_train,ligand_train,solvent_train = self.train_df['Catalyst'].to_list(), \
                                                             self.train_df['Base'].to_list(), \
                                                             self.train_df['Ligand'].to_list(),\
                                                             self.train_df['Solvent'].to_list()
      catalyst_train = [item for items in catalyst_train for item in items]
      base_train = [item for items in base_train for item in items]
      catalyst_test,base_test,ligand_test,solvent_test = self.test_df['Catalyst'].to_list(), \
                                                             self.test_df['Base'].to_list(), \
                                                             self.test_df['Ligand'].to_list(),\
                                                             self.test_df['Solvent'].to_list()
      catalyst_test = [item for items in catalyst_test for item in items]
      base_test = [item for items in base_test for item in items]
      catalyst_train,catalyst_test = np.array(catalyst_train).reshape(-1,1),np.array(catalyst_test).reshape(-1,1)
      base_train,base_test = np.array(base_train).reshape(-1,1),np.array(base_test).reshape(-1,1)
   
      # generate common features (conditions)
      catalyst_ohe,base_ohe,solvent_mhe = encode_conditions(catalyst_train,base_train,solvent_train)
      catalyst_train,base_train,solvent_train = catalyst_ohe.transform(catalyst_train).toarray(),base_ohe.transform(base_train).toarray(),solvent_mhe.transform(solvent_train)
      catalyst_test,base_test,solvent_test = catalyst_ohe.transform(catalyst_test).toarray(),base_ohe.transform(base_test).toarray(),solvent_mhe.transform(solvent_test)

      #validation set
      if self.model_type == 'nn':
         yield_val = self.val_df['Percent_Yield']
         self.y_val = get_yield_data(yield_val, self.task_type)

         boronic_acid_val = self.val_df['Boronic_Acid'].to_list()
         aryl_halide_val = self.val_df['Aryl_Halide'].to_list()
      
         catalyst_val,base_val,ligand_val,solvent_val = self.val_df['Catalyst'].to_list(), \
                                                             self.val_df['Base'].to_list(), \
                                                             self.val_df['Ligand'].to_list(),\
                                                             self.val_df['Solvent'].to_list()
         catalyst_val = [item for items in catalyst_val for item in items]
         base_val = [item for items in base_val for item in items]
         catalyst_val = np.array(catalyst_val).reshape(-1,1)
         base_val = np.array(base_val).reshape(-1,1)
         catalyst_val,base_val,solvent_val = catalyst_ohe.transform(catalyst_val).toarray(),base_ohe.transform(base_val).toarray(),solvent_mhe.transform(solvent_val)

      # generate desired features and concatenate
      radius, nBits = 2, 2048

      if self.feature_type == 'ohe':
         boronic_acid_train, boronic_acid_test = np.array(boronic_acid_train).reshape(-1,1), np.array(boronic_acid_test).reshape(-1,1)
         aryl_halide_train, aryl_halide_test = np.array(aryl_halide_train).reshape(-1,1), np.array(aryl_halide_test).reshape(-1,1)
         boronic_acid_ohe, aryl_halide_ohe = one_hot_encode(boronic_acid_train, aryl_halide_train)
         boronic_acid_ohe_train, aryl_halide_ohe_train = boronic_acid_ohe.transform(boronic_acid_train).toarray(), \
                                                         aryl_halide_ohe.transform(aryl_halide_train).toarray()
         boronic_acid_ohe_test,aryl_halide_ohe_test = boronic_acid_ohe.transform(boronic_acid_test).toarray(),aryl_halide_ohe.transform(aryl_halide_test).toarray()
         self.X_train = np.hstack((boronic_acid_ohe_train, aryl_halide_ohe_train, catalyst_train, base_train, solvent_train))
         if self.model_type == 'nn': 
            boronic_acid_val = np.array(boronic_acid_val).reshape(-1,1)
            aryl_halide_val = np.array(aryl_halide_val).reshape(-1,1)
            boronic_acid_ohe_val, aryl_halide_ohe_val = boronic_acid_ohe.transform(boronic_acid_val).toarray(), \
                                                         aryl_halide_ohe.transform(aryl_halide_val).toarray()
            self.X_val = np.hstack((boronic_acid_ohe_val, aryl_halide_ohe_val, catalyst_val, base_val, solvent_val))
         self.X_test = np.hstack((boronic_acid_ohe_test, aryl_halide_ohe_test, catalyst_test,base_test,solvent_test))

      elif self.feature_type == 'fgp':
         fgp_train = get_fgp_data(boronic_acid_train, aryl_halide_train, radius, nBits)
         fgp_test = get_fgp_data(boronic_acid_test, aryl_halide_test, radius, nBits)
         self.X_train = np.hstack((fgp_train, catalyst_train, base_train, solvent_train))
         if self.model_type == 'nn': 
            fgp_val = get_fgp_data(boronic_acid_val, aryl_halide_val, radius, nBits)
            self.X_val = np.hstack((fgp_val, catalyst_val, base_val, solvent_val))
         self.X_test = np.hstack((fgp_test, catalyst_test, base_test, solvent_test))

      elif self.feature_type == 'dft':
         dft_train = get_reactive_site_dft_data(self.train_df)
         dft_test = get_reactive_site_dft_data(self.test_df)
         scaler = MinMaxScaler().fit(dft_train)
         dft_train = scaler.transform(dft_train)
         dft_test = scaler.transform(dft_test)
         self.X_train = np.hstack((dft_train, catalyst_train, base_train, solvent_train))
         if self.model_type == 'nn':
            dft_val = get_reactive_site_dft_data(self.val_df)
            dft_val = scaler.transform(dft_val)
            self.X_val = np.hstack((dft_val, catalyst_val, base_val, solvent_val))
         self.X_test = np.hstack((dft_test, catalyst_test,base_test,solvent_test))

      elif self.feature_type == 'fgpdft':
         fgp_train = get_fgp_data(boronic_acid_train, aryl_halide_train, radius, nBits)
         fgp_test = get_fgp_data(boronic_acid_test, aryl_halide_test, radius, nBits)
         dft_train = get_reactive_site_dft_data(self.train_df)
         dft_test = get_reactive_site_dft_data(self.test_df)
         scaler = MinMaxScaler().fit(dft_train)
         dft_train = scaler.transform(dft_train)
         dft_test = scaler.transform(dft_test)
         self.X_train = np.hstack((fgp_train, catalyst_train, base_train, solvent_train, dft_train))
         if self.model_type == 'nn':
            fgp_val = get_fgp_data(boronic_acid_val, aryl_halide_val, radius, nBits)
            dft_val = get_reactive_site_dft_data(self.val_df)
            dft_val = scaler.transform(dft_val)
            self.X_val = np.hstack((fgp_val, catalyst_val, base_val, solvent_val, dft_val))
         self.X_test = np.hstack((fgp_test, catalyst_test, base_test, solvent_test, dft_test))
   
   def initialize_model(self):
      self.model_seed = 42
      if self.model_type == 'rf' and self.task_type == 'reg': self.model = RandomForestRegressor(random_state=self.model_seed, **self.model_params)
      elif self.model_type == 'rf' and (self.task_type == 'bin' or self.task_type == 'mul'): self.model = RandomForestClassifier(random_state=self.model_seed, **self.model_params)
      elif self.model_type == 'xgb' and self.task_type == 'reg': self.model = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.model_seed, **self.model_params)
      elif self.model_type == 'xgb' and self.task_type == 'bin': self.model = xgb.XGBClassifier(objective='binary:logistic', random_state=self.model_seed, **self.model_params)
      elif self.model_type == 'xgb' and self.task_type == 'mul': self.model = xgb.XGBClassifier(objective = 'multi:softprob', random_state=self.model_seed, **self.model_params)
      elif self.model_type == 'nn': self.model = FFNN(self.task_type, self.model_params, self.model_seed, self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test)
   
   def train_model(self):
      if self.model_type == 'nn': self.model.fit()
      else: self.model.fit(self.X_train,self.y_train)
   
   def model_predict(self):
      if self.model_type == 'nn': self.y_pred = self.model.predict()
      else: self.y_pred = self.model.predict(self.X_test)
   
   def train_test_model(self):
      self.get_model_inputs()
      self.initialize_model()
      self.train_model()
      self.model_predict()
      if self.y_test is not None: 
         self.res = score_predictions(self.task_type, self.y_test, self.y_pred)