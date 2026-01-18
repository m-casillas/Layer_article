import joblib
from LayerRepresentation import LayerRepresentation
from Genotype import Genotype
from globalsENAS import *
import ast
from file_handler import *
from datetime import datetime

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
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    from sklearn.preprocessing import MinMaxScaler
    from scipy.stats import randint, uniform
    from scipy.stats import spearmanr, kendalltau
    from sklearn.metrics import mean_absolute_error, make_scorer

def spearman_scorer(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation

def kendall_scorer(y_true, y_pred):
    return kendalltau(y_true, y_pred).correlation

kendall_score = make_scorer(kendall_scorer, greater_is_better=True)
spearman_sklearn = make_scorer(spearman_scorer,  greater_is_better=True)

class Surrogate_ENAS:
    def train_all_surrogates(self):
        results = {}
        log_filename = os.path.join(self.regressor_folder,  f'{ConfigClass.REPRESENTATION_TYPE}regressor_training_results_{datetime.now().strftime("%Y-%m-%d_%H-%M")}.txt')
        #log_filename = os.path.join(self.regressor_folder, f"{ConfigClass.REPRESENTATION_TYPE}regressor_training_results.txt")

        with open(log_filename, "w") as f:
            for model_choice in self.list_of_regressors:
                self.encodings = []
                self.accuracy = []
                self.encoding_matrix = []
                self.X = []
                self.y = []
                self.model_choice = model_choice
                self.load_archs_CSV()
                #rmse, r2 = self.train_Surrogate_model()
                rmse, r2, mae, pearson, spearman, kendall = self.train_Surrogate_model()
                results[model_choice] = (rmse, r2, mae, pearson, spearman, kendall)

            header = "All models trained. Results:"
            print(header)
            
            header_line = f"{'Index':<20} {'Model':<20} {'RMSE':>10} {'R²':>10} {'MAE':>10} {'Pearson':>10} {'Spearman':>10} {'Kendall':>10}"
            print(header_line)
            f.write(header_line + "\n")

            separator = "-" * 125
            print(separator)
            f.write(separator + "\n")

            for model_choice, (rmse, r2, mae, pearson, spearman, kendall) in sorted(results.items(), key=lambda x: x[1][1], reverse=True):
                model_name = self.regressors_dict[model_choice]
                line = f"{model_choice:<20} {model_name:<20} {rmse:>10.4f} {r2:>10.4f} {mae:>10.4f} {pearson:>10.4f} {spearman:>10.4f} {kendall:>10.4f}"
                print(line)
                f.write(line + "\n")

        print(f"\nResults also saved to {log_filename}")
        print(f'Copying the CSV file to the surrogate folder')
        shutil.copy(self.archs_CSV, self.regressor_folder)

    def load_archs_CSV(self):
        print(f'Loading architectures for the Surrogate Model from {self.archs_CSV}')
        df = pd.read_csv(self.archs_CSV)
        df.dropna(subset=['FLOPs'], inplace=True)
        #df = df[df['BestGen'] == False]
        df = remove_duplicates(df, column_name='Integer_encoding', metric='Accuracy', keep='max')
        self.encodings = df['Integer_encoding'].apply(ast.literal_eval)  # Convert string list to actual list
        self.X = np.vstack(self.encodings)
        self.y = df['Accuracy'].values.astype(np.float32)
        print(f'Loading complete')
    
    def preprocess_data(self, X):
        print2('Processing data for the regressor')
        # === 4. One-hot encode categorical features ===
        self.categorical_features = list(range(X.shape[1]))
        self.preprocessor = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), self.categorical_features)
        ])
        print2('Processing complete\n')
        return X

    
    def predict_CSV(self, test_filename):
        self.load_archs_CSV(test_filename)
        X = self.preprocess_CSV(self.X)
        print2(f'\nPredicting CSV with regressor {self.regressors_dict[self.model_choice]}')
        print2(f'from {self.regressor_filepath}')
        model = joblib.load(self.regressor_filepath)
        y_pred = model.predict(X)
        print2('Prediction complete\n')

    def load_arch(self, arch):
            print2(f'Loading architecture {arch.idx} for predicting accuracy')
            self.encodings = arch.integer_encoding
            self.encoding_matrix = np.array(self.encodings, dtype=np.float32)
            self.X = pd.DataFrame(self.encoding_matrix.reshape(1, -1), columns=self.columns) #CHANGED THIS ONE
            #print(f'Load archs CSV {self.X.shape = } {self.y.shape = }')
            print2(f'Loading complete\n')

    def predict_arch(self, arch_idx = '', model_choice = 10):
        X = self.preprocess_data(self.X)
        print2(f'\nPredicting {arch_idx} with regressor {self.regressors_dict[model_choice]}')
        print2(f'from {self.regressor_filepath}')
        #model = joblib.load(self.regressor_filepath)
        y_pred = self.model.predict(X)
        print2('Prediction complete\n')
        return y_pred[0]
    
    

    def train_Surrogate_model(self): #Version with fine tuning
        print('\n============================================================================================================')
        print(f'Training {self.regressors_dict[self.model_choice]}')


       # Step 1: Split into train+val and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Preprocess data
        self.X = self.preprocess_data(self.X)
 
        #Define models ===
        self.model_map = {
            0: XGBRegressor(random_state=42, n_jobs=-1),
            1: RandomForestRegressor(random_state=42, n_jobs=-1),
            10: GradientBoostingRegressor(random_state=42)
        }
 
        self.param_spaces = {
            0: {
                "model__n_estimators": randint(300, 800),
                "model__learning_rate": uniform(0.01, 0.2),
                "model__max_depth": randint(3, 10),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.7, 0.3)
            },
            1: {
                "model__n_estimators": randint(200, 600),
                "model__max_depth": randint(5, 20),
                "model__min_samples_split": randint(2, 10),
                "model__min_samples_leaf": randint(1, 5)
            },
            10: {
                "model__n_estimators": randint(200, 600),
                "model__learning_rate": uniform(0.01, 0.2),
                "model__max_depth": randint(3, 10),
                "model__subsample": uniform(0.7, 0.3)
            }
            
        }

        if self.model_choice not in self.model_map:
            raise ValueError(f"Invalid model_choice {self.model_choice}")

        self.model = self.model_map[self.model_choice]
        self.results = {}
        self.pipe = Pipeline([('preprocessor', self.preprocessor), ('model', self.model)])
        self.search = RandomizedSearchCV(
            estimator = self.pipe,
            param_distributions = self.param_spaces[self.model_choice],
            n_iter = 20,
            cv = 3,
            scoring = 'r2', # 'r2', #spearman_sklearn, #kendall_score, neg_mean_absolute_error
            verbose = 1, 
            random_state = 42,
            n_jobs = 1
        )

        print(f'Training {self.regressors_dict[self.model_choice]}')
        self.search.fit(X_train, y_train)
        self.best_model = self.search.best_estimator_
        y_pred = self.best_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mae = mean_absolute_error(y_test, y_pred)  # Add this
        pearson = np.corrcoef(y_test, y_pred)[0,1]
        spearman, _ = spearmanr(y_test, y_pred)
        kendall, _ = kendalltau(y_test, y_pred)
      
        # Save model
        ensure_folder_exists(self.regressor_folder)
        self.regressor_path = os.path.join(self.regressor_folder, f'regressor_{self.model_choice}.joblib')
        if self.SAVE_MODELS:
            joblib.dump(self.best_model, self.regressor_path)
            print(f'Training {self.regressors_dict[self.model_choice]} complete. Model saved as {self.regressor_path}')
        else:
            print(f'Training {self.regressors_dict[self.model_choice]} complete.')
        print('============================================================================================================\n')

        # Return validation metrics
        return rmse, r2, mae, pearson, spearman, kendall
        
    
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
        self.regressors_dict = {0: 'XGBoost', 1: 'Random Forest', 10: 'Gradient Boosting'}
        self.list_of_regressors = [0, 1, 10]
        if TRAINING_NEW_MODELS == False:
            self.model = joblib.load(self.regressor_filepath)

os.system("cls")



