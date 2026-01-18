import joblib
from LayerRepresentation import LayerRepresentation
from Genotype import Genotype
from globalsENAS import *
import ast
from file_handler import *

if ConfigClass.SURROGATE == True:
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
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split, GridSearchCV  # Added GridSearchCV
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    from sklearn.preprocessing import MinMaxScaler





class Surrogate_ENAS:
    def train_all_surrogates(self):
        results = {}
        log_filename = os.path.join(self.regressor_folder, f"{ConfigClass.REPRESENTATION_TYPE}regressor_training_results.txt")

        with open(log_filename, "w") as f:
            for model_choice in self.list_of_regressors:
                self.encodings = []
                self.accuracy = []
                self.encoding_matrix = []
                self.X = []
                self.y = []
                self.model_choice = model_choice
                self.load_archs_CSV()
                rmse, r2 = self.train_Surrogate_model()
                results[model_choice] = (rmse, r2)

            header = "All models trained. Results:"
            print(header)
            
            header_line = f"{'Index':<20} {'Model':<20} {'RMSE':>10} {'R²':>10}"
            print(header_line)
            f.write(header_line + "\n")

            separator = "-" * 60
            print(separator)
            f.write(separator + "\n")

            for model_choice, (rmse, r2) in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
                model_name = self.regressors_dict[model_choice]
                line = f"{model_choice:<20} {model_name:<20} {rmse:>10.4f} {r2:>10.4f}"
                print(line)
                f.write(line + "\n")

        print(f"\nResults also saved to {log_filename}")
        print(f'Copying the CSV file to the surrogate folder')
        shutil.copy(self.archs_CSV, self.regressor_folder)

    def load_archs_CSV(self):
        print(f'Loading architectures for the Surrogate Model from {self.archs_CSV}')
        df = pd.read_csv(self.archs_CSV)
        #df = df[df['BestGen'] == False]
        self.encodings = df['Integer_encoding'].apply(ast.literal_eval)  # Convert string list to actual list
        self.encoding_matrix = np.array(self.encodings.tolist(), dtype=np.float32)
        self.y = df['Accuracy'].values.astype(np.float32)
        self.X = self.encoding_matrix
        self.X = pd.DataFrame(self.X, columns=self.columns)
        self.X = self.X.astype('category')
        self.y = pd.DataFrame(self.y, columns=['Accuracy'])
        print(f'Loading complete')
    
    def preprocess_data(self, X):
        print('Processing data for the regressor')
        columns = X.columns.tolist()
        #nominal_cols = [col for col in columns if col.startswith('N')]
        # One-hot encode nominal variables
        #X = pd.get_dummies(X[columns], drop_first=False)
        X = pd.get_dummies(X[columns], columns=columns, drop_first=False)
        #X = X[columns]
        print('Processing complete\n')
        return X

    
    def predict_CSV(self, test_filename):
        self.load_archs_CSV(test_filename)
        X = self.preprocess_CSV(self.X)
        print(f'\nPredicting CSV with regressor {self.regressors_dict[self.model_choice]}')
        print(f'from {self.regressor_filepath}')
        model = joblib.load(self.regressor_filepath)
        y_pred = model.predict(X)
        print(self.y)
        print(y_pred)
        print('Prediction complete\n')

    def load_arch(self, arch):
            print(f'Loading architecture {arch.idx} for predicting accuracy')
            self.encodings = arch.integer_encoding
            self.encoding_matrix = np.array(self.encodings, dtype=np.float32)
            self.X = pd.DataFrame(self.encoding_matrix.reshape(1, -1), columns=self.columns) #CHANGED THIS ONE
            #print(f'Load archs CSV {self.X.shape = } {self.y.shape = }')
            print(f'Loading complete\n')

    def predict_arch(self, arch_idx = '', model_choice = 10):
        X = self.preprocess_data(self.X)
        print(f'\nPredicting {arch_idx} with regressor {self.regressors_dict[model_choice]}')
        print(f'from {self.regressor_filepath}')
        #model = joblib.load(self.regressor_filepath)
        y_pred = self.model.predict(X)
        print('Prediction complete\n')
        return y_pred[0]
    
    

    def train_Surrogate_model(self): #Version with fine tuning and validation acc
        print('\n============================================================================================================')
        print(f'Training {self.regressors_dict[self.model_choice]}')

        # Preprocess data
        self.X = self.preprocess_data(self.X)
        # Step 1: Split into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Step 2: Split train+val into train and validation sets (0.25 * 0.8 = 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42
        )

        # Model mapping. Only keep forest based models. MLP, Lasso, Huber, etc. do not work well
        model_map = {
            0: XGBRegressor(objective='reg:squarederror', enable_categorical=True),
            1: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            2: Ridge(alpha=1.0),
            3: SVR(kernel='rbf', C=1.0, epsilon=0.1),
            4: MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42),
            5: LinearRegression(),
            6: Lasso(alpha=0.1),
            7: ElasticNet(alpha=0.1, l1_ratio=0.5),
            8: BayesianRidge(),
            9: HuberRegressor(),
            10: GradientBoostingRegressor(random_state=42),
            11: AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            12: ExtraTreesRegressor(n_estimators=100, random_state=42),
            13: DecisionTreeRegressor(max_depth=10, random_state=42),
            14: KNeighborsRegressor(n_neighbors=5),
            15: GaussianProcessRegressor(kernel=RBF(), random_state=42)
        }

        if self.model_choice not in model_map:
            raise ValueError(f"Invalid model_choice {self.model_choice}. Choose from 0 to {len(model_map) - 1}.")

        self.model = model_map[self.model_choice]

        # Fine-tuning for GradientBoostingRegressor (model_choice=10) and XGBRegressor (model_choice=0)
        if self.model_choice in [0, 10]:
            param_grid = {
                0: {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'min_child_weight': [1, 3, 5]  # Specific to XGBRegressor
                },
                10: {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            }[self.model_choice]

            print(f"Performing GridSearchCV for {self.regressors_dict[self.model_choice]}")
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=5,  # 5-fold cross-validation
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)

            # Update model with best parameters
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        else:
            # Train non-tuned models directly
            self.model.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        print(f"Validation RMSE: {val_rmse:.4f}, Validation R²: {val_r2:.4f}")

        # Evaluate on test set
        y_test_pred = self.model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        print(f"Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")

        # Save model
        ensure_folder_exists(self.regressor_folder)
        self.regressor_path = os.path.join(self.regressor_folder, f'regressor_{self.model_choice}.joblib')
        if self.SAVE_MODELS:
            joblib.dump(self.model, self.regressor_path)
            print(f'Training {self.regressors_dict[self.model_choice]} complete. Model saved as {self.regressor_path}')
        else:
            print(f'Training {self.regressors_dict[self.model_choice]} complete.')
        print('============================================================================================================\n')

        # Return validation metrics
        return val_rmse, val_r2
        
    def train_Surrogate_model_nofine(self): #Version with no fine tuning
        print('\n============================================================================================================')
        print(f'Training {self.regressors_dict[self.model_choice]}')

        # Preprocess data
        self.X = self.preprocess_data(self.X)

        # Step 1: Split into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Step 2: Split train+val into train and validation sets (0.25 * 0.8 = 0.2)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42
        )

        # Model mapping
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
            10: GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42),
            11: AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            12: ExtraTreesRegressor(n_estimators=100, random_state=42),
            13: DecisionTreeRegressor(max_depth=10, random_state=42),
            14: KNeighborsRegressor(n_neighbors=5),
            15: GaussianProcessRegressor(kernel=RBF(), random_state=42)
        }

        if self.model_choice not in model_map:
            raise ValueError(f"Invalid model_choice {self.model_choice}. Choose from 0 to {len(model_map) - 1}.")

        self.model = model_map[self.model_choice]

        # Train on train set
        self.model.fit(X_train, y_train)

        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        print(f"Validation RMSE: {val_rmse:.4f}, Validation R²: {val_r2:.4f}")

        # Evaluate on test set
        y_test_pred = self.model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        print(f"Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")

        # Save model
        ensure_folder_exists(self.regressor_folder)
        self.regressor_path = os.path.join(self.regressor_folder, f'regressor_{self.model_choice}.joblib')
        if self.SAVE_MODELS:
            joblib.dump(self.model, self.regressor_path)
            print(f'Training {self.regressors_dict[self.model_choice]} complete. Model saved as {self.regressor_path}')
        else:
            print(f'Training {self.regressors_dict[self.model_choice]} complete.')
        print('============================================================================================================\n')

        # Return validation metrics to use for tuning if needed
        return val_rmse, val_r2

    def __init__(self, model_choice, regressor_folder = '', archs_CSV = '', SAVE_MODELS = False, TRAINING_NEW_MODELS = False):
        if ConfigClass.REPRESENTATION_TYPE == 'L':
            self.columns = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7']
        elif ConfigClass.REPRESENTATION_TYPE == 'B':
            self.columns = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8']
        self.SAVE_MODELS = SAVE_MODELS
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
        ensure_folder_exists(self.regressor_folder)
        self.regressor_filename = 'regressor_' + str(self.model_choice) + '.joblib'
        self.regressor_filepath = os.path.join(self.regressor_folder, self.regressor_filename)
        self.regressors_dict = {0: 'XGBoost', 1: 'Random Forest', 2: 'Ridge', 3: 'SVR', 4: 'MLP', 5: 'Linear Regression', 6: 'Lasso', 7: 'ElasticNet', 8: 'Bayesian Ridge', 9: 'Huber', 
                                10: 'Gradient Boosting', 11: 'AdaBoost', 12: 'Extra Trees', 13: 'Decision Tree', 14: 'KNN', 15: 'Gaussian Process'}
        self.list_of_regressors = list(range(15))# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        if TRAINING_NEW_MODELS == False:
            self.model = joblib.load(self.regressor_filepath)

os.system("cls")



