from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import os
import ast 
import numpy as np
import pandas as pd
import joblib
from LayerRepresentation import LayerRepresentation
from Genotype import Genotype

class Surrogate_ENAS:
    def train_all_surrogates(self):
        results = {}
        for model_choice in self.list_of_regressors:
        #for model_choice in self.regressors_dict.keys():
            self.encodings = []
            self.flops = []
            self.num_params = []
            self.accuracy = []
            self.encoding_matrix = []
            self.X = []
            self.y = []
            self.model_choice = model_choice
            self.load_archs_CSV()
            rmse, r2 = self.train_Surrogate_model()
            results[model_choice] = (rmse, r2)
            #self.predict_CSV(self.test_filename)
       
        print("All models trained. Results:")
        print(f"{'Model':<20} {'RMSE':>10} {'R²':>10}")
        print("-" * 42)

        for model_choice, (rmse, r2) in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
            model_name = self.regressors_dict[model_choice]
            print(f"{model_name:<20} {rmse:>10.4f} {r2:>10.4f}")

    def load_archs_CSV(self):
        print(f'Loading architectures for the Surrogate Model from {self.archs_CSV}')
        df = pd.read_csv(self.archs_CSV)
        df = df[df['BestGen'] == False]
        self.encodings = df['Integer_encoding'].apply(ast.literal_eval)  # Convert string list to actual list
        self.encoding_matrix = np.array(self.encodings.tolist(), dtype=np.float32)
        self.flops = df['FLOPs'].values.reshape(-1, 1)
        self.num_params = df['Num_Params'].values.reshape(-1, 1)
        self.X = np.hstack([self.encoding_matrix, self.flops, self.num_params])
        self.y = df['Accuracy'].values.astype(np.float32)
        columns = ['N1','C11','C12','N2','C21','C22','N3','C31','C32','N4','C41','C42','N5','C51','C52','N6','C61','C62','N7','C71','C72','R1', 'R2']
        self.X = pd.DataFrame(self.X, columns=columns)
        self.y = pd.DataFrame(self.y, columns=['Accuracy'])
        #print(f'Load archs CSV {self.X.shape = } {self.y.shape = }')
        print(f'Loading complete')
    
    def preprocess_CSV(self, X):
        print('Processing data before training the regressor')
        columns = X.columns.tolist()
        nominal_cols = [col for col in columns if col.startswith('N')]
        ordinal_cols = [col for col in columns if col.startswith('C')]
        real_cols =    [col for col in columns if col.startswith('R')]
       
        # One-hot encode nominal variables
        X_nominal = pd.get_dummies(X[nominal_cols], drop_first=True)
        X_ordinal = X[ordinal_cols].copy()
        # Standardize real variables
        scaler = StandardScaler()
        X[real_cols] = scaler.fit_transform(X[real_cols])

        # Combine features
        X = pd.concat([X[real_cols], X_ordinal, X_nominal], axis=1)
        #X = pd.concat([X_ordinal, X_nominal], axis=1)
        print('Processing complete\n')
        return X
    
    def predict_CSV(self, test_filename):
        self.load_archs_CSV(test_filename)
        X = self.preprocess_CSV(self.X)
        print(f'\nPredicting CSV with regressor {self.regressors_dict[self.model_choice]}')
        # Load model (same method for all models)
        model = joblib.load(self.regressor_filepath)
        y_pred = model.predict(X)
        print(self.y)
        print(y_pred)
        print('Prediction complete\n')

    def load_arch(self, arch):
            print(f'Loading architecture {arch.idx} for predicting accuracy')
            self.encodings = arch.integer_encoding
            self.encoding_matrix = np.array(self.encodings, dtype=np.float32)
            self.flops = arch.flops
            self.num_params = arch.num_params
            self.X = np.hstack([self.encoding_matrix, self.flops, self.num_params])
            #N for nominal, like CONV, POOLMAX. C for categorical, like 0, 1, 2. R for real, like FLOPs, Num_Params
            columns = ['N1','C11','C12','N2','C21','C22','N3','C31','C32','N4','C41','C42','N5','C51','C52','N6','C61','C62','N7','C71','C72','R1', 'R2']
            self.X = pd.DataFrame(self.X.reshape(1, -1), columns=columns)
            #print(f'Load archs CSV {self.X.shape = } {self.y.shape = }')
            print(f'Loading complete\n')

    def predict_arch(self, arch, model_choice):
        X = self.preprocess_CSV(self.X)
        print(f'\nPredicting {arch.idx} with regressor {self.regressors_dict[model_choice]}')
        model = joblib.load(self.regressor_filepath)
        y_pred = model.predict(X)
        print('Prediction complete\n')
        return y_pred[0]
        
    def train_Surrogate_model(self):
        print('\n============================================================================================================')
        print(f'Training {self.regressors_dict[self.model_choice]}')
        self.X = self.preprocess_CSV(self.X)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Select model based on model_choice
        model_map = {
            0: XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5),
            1: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            2: Ridge(alpha=1.0),
            3: SVR(kernel='rbf', C=1.0, epsilon=0.1),
            4: MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42),
            5: LinearRegression(),
            6: Lasso(alpha=0.1),
            7: ElasticNet(alpha=0.1, l1_ratio=0.5),
            8: BayesianRidge(),
            9: HuberRegressor(),
            10: GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            11: AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            12: ExtraTreesRegressor(n_estimators=100, random_state=42),
            13: DecisionTreeRegressor(max_depth=10, random_state=42),
            14: KNeighborsRegressor(n_neighbors=5),
            15: GaussianProcessRegressor(kernel=RBF(), random_state=42)
        }

        if self.model_choice not in model_map:
            raise ValueError(f"Invalid model_choice {self.model_choice}. Choose from 0 to {len(model_map) - 1}.")

        model = model_map[self.model_choice]

        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")

        # Save model
        self.regressor_path = os.path.join(self.regressor_folder, f'regressor_{self.model_choice}.joblib') 
        if SAVE_MODELS == True:
            joblib.dump(model, self.regressor_path)
            print(f'Training {self.regressors_dict[self.model_choice]} complete. Model saved as {self.regressor_path}')
        else:
            print(f'Training {self.regressors_dict[self.model_choice]} complete.')
        print('============================================================================================================\n')
        return rmse, r2

    def __init__(self, model_choice = 0, regressor_folder = '', archs_CSV = ''):
        self.model_choice = model_choice
        self.encodings = []
        self.flops = []
        self.num_params = []
        self.accuracy = []
        self.encoding_matrix = []
        self.X = []
        self.y = []
        self.archs_CSV = archs_CSV
        self.regressor_folder = regressor_folder
        self.regressor_filename = 'regressor_' + str(self.model_choice) + '.joblib'
        self.regressor_filepath = os.path.join(self.regressor_folder, self.regressor_filename)
        #self.regressors_dict = {0:'XGBoost', 1:'Random Forest', 2:'Ridge', 3:'SVR', 4:'MLP'}
        self.regressors_dict = {0: 'XGBoost', 1: 'Random Forest', 2: 'Ridge', 3: 'SVR', 4: 'MLP', 5: 'Linear Regression', 6: 'Lasso', 7: 'ElasticNet', 8: 'Bayesian Ridge', 9: 'Huber', 
                                10: 'Gradient Boosting', 11: 'AdaBoost', 12: 'Extra Trees', 13: 'Decision Tree', 14: 'KNN', 15: 'Gaussian Process'}
        self.list_of_regressors = list(range(16))# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


os.system("cls")
regressor_folder = r'C:\Users\xaero\OneDrive\ITESM DCC\Layer_article\results\surrogates'
archs_CSV= os.path.join(regressor_folder, 'merged.csv')
SAVE_MODELS = True
surrogate = Surrogate_ENAS(regressor_folder = regressor_folder, archs_CSV = archs_CSV)
surrogate.train_all_surrogates()

