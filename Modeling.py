
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
import joblib  # Model serialization

from sklearn.preprocessing import StandardScaler, LabelEncoder  # Data preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold  # Model selection and cross-validation
from sklearn.pipeline import Pipeline  # Pipeline creation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  # Model evaluation metrics

from sklearn.tree import DecisionTreeClassifier  # Decision Tree algorithm
from sklearn.linear_model import LogisticRegression, SGDClassifier  # Linear models
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier, 
    BaggingClassifier, 
    StackingClassifier,
    VotingClassifier  # Ensemble methods
)
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.kernel_approximation import Nystroem  # Kernel approximation

import xgboost as xgb  # XGBoost library
from xgboost import XGBClassifier  # XGBoost classifier
import lightgbm as lgb  # LightGBM library
from lightgbm import LGBMClassifier  # LightGBM classifier
from catboost import CatBoostClassifier  # CatBoost classifier



# Load and optimize datasets with data type correction
def load():
    train_data = pd.read_csv("Data/train_preprocessed.csv", encoding='utf-8', na_values=['', ' ', 'NA'])
    test_data = pd.read_csv("Data/test_preprocessed.csv", encoding='utf-8', na_values=['', ' ', 'NA'])
    
    # Identify columns that should be categorical
    categorical_cols = [
        'Gender', 'Customer_Type', 'Type_Travel', 'Travel_Class', 'Seat_Class',
        'Platform_Location', 'Age_Group', 'Onboard_Wifi_Service', 
        'Ease_of_Online_Booking', 'Online_Support', 'Legroom', 'Cleanliness',
        "Online_Boarding","CheckIn_Service","Baggage_Handling","Onboard_Service",
        "Onboard_Entertainment","Catering","Arrival_Time_Convenient","Seat_Comfort"
    ]

    # Ensure specified columns are converted to category type
    for col in categorical_cols:
        if col in train_data.columns:
            train_data[col] = train_data[col].astype('category')
        if col in test_data.columns:
            test_data[col] = test_data[col].astype('category')
    
    # Convert specific columns to optimized data types
    float_cols = ['Age', 'Travel_Distance', 'Departure_Delay_in_Mins', 
                  'Arrival_Delay_in_Mins', 'Log_Travel_Distance', 
                  'Log_Departure_Delay_in_Mins', 'Log_Arrival_Delay_in_Mins']
    
    int_cols = ['Overall_Experience']
    
    # Convert float64 to float32
    for col in float_cols:
        if col in train_data.columns:
            train_data[col] = train_data[col].astype('float32')
        if col in test_data.columns:
            test_data[col] = test_data[col].astype('float32')
    
    # Convert int64 to int32
    for col in int_cols:
        if col in train_data.columns:
            train_data[col] = train_data[col].astype('int32')
        if col in test_data.columns:
            test_data[col] = test_data[col].astype('int32')

    print("Data loaded and optimized for memory usage with corrected data types.")
    print(train_data.info())
    return train_data, test_data

# Step 1: Change 'ID' column type to string
def ID_Change(train_data,test_data):
    train_data['ID'] = train_data['ID'].astype(str)
    test_data['ID'] = test_data['ID'].astype(str)

    # Verify the change
    print("'ID' column type in train_data:", train_data['ID'].dtype)
    print("'ID' column type in test_data:", test_data['ID'].dtype)

    return train_data,test_data

# Step 2: Convert Age to integer if it doesn’t have decimal points
def float_to_int_for_memory_management(train_data,test_data):
    if train_data['Age'].dropna().apply(float.is_integer).all():
        train_data['Age'] = train_data['Age'].astype(int)
    if test_data['Age'].dropna().apply(float.is_integer).all():
        test_data['Age'] = test_data['Age'].astype(int)

    # Convert delay columns to integer if they don’t have decimal points
    if train_data['Departure_Delay_in_Mins'].dropna().apply(float.is_integer).all():
        train_data['Departure_Delay_in_Mins'] = train_data['Departure_Delay_in_Mins'].astype(int)
    if test_data['Departure_Delay_in_Mins'].dropna().apply(float.is_integer).all():
        test_data['Departure_Delay_in_Mins'] = test_data['Departure_Delay_in_Mins'].astype(int)

    if train_data['Arrival_Delay_in_Mins'].dropna().apply(float.is_integer).all():
        train_data['Arrival_Delay_in_Mins'] = train_data['Arrival_Delay_in_Mins'].astype(int)
    if test_data['Arrival_Delay_in_Mins'].dropna().apply(float.is_integer).all():
        test_data['Arrival_Delay_in_Mins'] = test_data['Arrival_Delay_in_Mins'].astype(int)

    return train_data,test_data

# Step 3: Interaction Feature Between Delays
def Interaction_Feature(train_data,test_data):
    
    train_data['Total_Delay'] = train_data['Departure_Delay_in_Mins'] + train_data['Arrival_Delay_in_Mins']
    test_data['Total_Delay'] = test_data['Departure_Delay_in_Mins'] + test_data['Arrival_Delay_in_Mins']
    print("Interaction feature 'Total_Delay' created.")
    return train_data,test_data

# Step 4: Combining Satisfaction Indicators into a Composite Score
def combining_satisfaction_features(train_data, test_data):
    satisfaction_columns = [
        'Seat_Comfort', 'Arrival_Time_Convenient', 'Onboard_Wifi_Service', 'Onboard_Entertainment',
        'Catering', 'Legroom', 'Baggage_Handling', 'CheckIn_Service',
        'Cleanliness', 'Onboard_Service'
    ]
    online_satisfaction_columns = ['Online_Boarding', 'Online_Support', 'Ease_of_Online_Booking']

    # Define ordinal mapping for satisfaction categories
    satisfaction_mapping = {
        "Extremely Poor": 1,
        "Poor": 2,
        "Needs Improvement": 3,
        "Acceptable": 4,
        "Good": 5,
        "Excellent": 6
    }
    

    # Apply the mapping to the satisfaction and online satisfaction features
    for feature in satisfaction_columns + online_satisfaction_columns:
        train_data[feature] = train_data[feature].map(satisfaction_mapping).astype('float')
        test_data[feature] = test_data[feature].map(satisfaction_mapping).astype('float')

    # Calculate the satisfaction score as the mean of these ordinal values
    train_data['combining_satisfaction_score'] = train_data[satisfaction_columns].mean(axis=1)
    test_data['combining_satisfaction_score'] = test_data[satisfaction_columns].mean(axis=1)

    # Calculate the online satisfaction score as the mean of these ordinal values
    train_data['Online_Satisfaction_Score'] = train_data[online_satisfaction_columns].mean(axis=1)
    test_data['Online_Satisfaction_Score'] = test_data[online_satisfaction_columns].mean(axis=1)

    print("satisfaction_features scores (online and combining) created.")
    return train_data, test_data

# Step 5: One-Hot Encode Remaining Categorical Data
def encode_categorical_features(train_data, test_data):
    le = LabelEncoder()
    train_data["Platform_Location"] = le.fit_transform(train_data["Platform_Location"])
    test_data["Platform_Location"] = le.transform(test_data["Platform_Location"])
    # List of categorical columns to be one-hot encoded
    categorical_columns = [
        'Age_Group', 'Seat_Class', 'Travel_Class',
        'Type_Travel', 'Customer_Type', 'Gender'
    ]

    # One-hot encode the categorical columns in the training and test data
    train_data = pd.get_dummies(train_data, columns=categorical_columns, drop_first=True)
    test_data = pd.get_dummies(test_data, columns=categorical_columns, drop_first=True)

    # Align the train and test data to ensure they have the same columns after encoding
    # train_data, test_data = train_data.align(test_data, join='inner', axis=1)

    print("Categorical features one-hot encoded.")
    return train_data, test_data

# step 6 : modeling
def logistic_regression_model(X_train, y_train, X_valid, y_valid, optimize=False):
    print("################### Logistic Regression ###############")

    # from sklearn.preprocessing import PolynomialFeatures
    # poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # X_train_poly = poly.fit_transform(X_train)
    # X_valid_poly = poly.transform(X_valid)
    
    # Default best parameters if grid search is not used
    default_params = {'C': 0.01}
    
    # Initialize model with or without grid search
    if optimize:
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['elasticnet'],
            'l1_ratio': [0.2, 0.5, 0.8]
        }
        grid_search = GridSearchCV(
            LogisticRegression(max_iter=5000, solver='saga', n_jobs=-1),
            param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Logistic Regression - Best Params from GridSearch: {grid_search.best_params_}")
    else:
        # Use default best params
        best_model = LogisticRegression(C=default_params['C'], max_iter=5000, solver='saga', n_jobs=-1)
        best_model.fit(X_train, y_train)
        print(f"Logistic Regression - Using Default Params: {default_params}")
    
    # Make predictions
    y_pred = best_model.predict(X_valid)
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': roc_auc_score(y_valid, y_pred_proba)
    }

    # Save the model
    joblib.dump(best_model, 'models/logistic_regression_model.pkl')
    
    print("Logistic Regression Metrics:", metrics)
    return metrics

def decision_tree_model(X_train, y_train, X_valid, y_valid, optimize=False):
    print("################### Decision Tree ###############")
    
    # Default best parameters if grid search is not used
    default_params = {'max_depth': 15, 'min_samples_split': 75 ,'max_leaf_nodes':100 ,'criterion': 'entropy'}
    
    # Initialize model with or without grid search
    if optimize:
        param_grid = {
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10, 20 , 75],
            'max_leaf_nodes': [20, 50, 100],
            'criterion': ['gini', 'entropy']
        }
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Decision Tree - Best Params from GridSearch: {grid_search.best_params_}")
    else:
        # Use default best params
        best_model = DecisionTreeClassifier(max_depth=default_params['max_depth'], min_samples_split=default_params['min_samples_split'])
        best_model.fit(X_train, y_train)
        print(f"Decision Tree - Using Default Params: {default_params}")
    
    # Make predictions
    y_pred = best_model.predict(X_valid)
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': roc_auc_score(y_valid, y_pred_proba)
    }

    # Save the model
    joblib.dump(best_model, 'models/decision_tree_model.pkl')
    
    print("Decision Tree Metrics:", metrics)
    return metrics

def random_forest_model(X_train, y_train, X_valid, y_valid, optimize=False):
    print("################### Random Forest ###############")
    
    # Default best parameters if grid search is not used
    default_params = {'n_estimators': 500, 'max_depth': 25, 'min_samples_split': 5 ,'max_features': 'sqrt'}
    
    # Initialize model with or without grid search
    if optimize:
        param_grid = {
            'n_estimators': [220, 300, 500],
            'max_depth': [15, 20, 25],
            'min_samples_split': [5, 10],
            'max_features': ['sqrt', 'log2']
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42,n_jobs=-1),
            param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"Random Forest - Best Params from GridSearch: {grid_search.best_params_}")
    else:
        # Use default best params
        best_model = RandomForestClassifier(
            n_estimators=default_params['n_estimators'],
            max_depth=default_params['max_depth'],
            min_samples_split=default_params['min_samples_split'],
            n_jobs=-1
        )
        best_model.fit(X_train, y_train)
        print(f"Random Forest - Using Default Params: {default_params}")
    
    # Make predictions
    y_pred = best_model.predict(X_valid)
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': roc_auc_score(y_valid, y_pred_proba)
    }

    # Save the model
    joblib.dump(best_model, 'models/random_forest_model.pkl')
    
    print("Random Forest Metrics:", metrics)
    return metrics

############################# Not used ###############################
def svm_model(X_train, y_train, X_valid, y_valid):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    grid_search = GridSearchCV(SVC(probability=True),
                               param_grid, cv=StratifiedKFold(n_splits=3), scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    # Save the model
    joblib.dump(best_model, 'svm_model.pkl')
    
    y_pred = best_model.predict(X_valid)
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': roc_auc_score(y_valid, y_pred_proba)
    }
    print(f"SVM - Best Params: {grid_search.best_params_}")
    return metrics

def nystroem_svm_model(X_train, y_train, X_valid, y_valid):
    # Set up the pipeline with Nystroem and SVC
    nystroem_approx = Nystroem(kernel='rbf', n_components=500, random_state=42)  # Adjust n_components for approximation accuracy
    svm = SVC(kernel='linear', probability=True)

    pipeline = Pipeline([
        ('feature_map', nystroem_approx),
        ('svm', svm)
    ])
    joblib.dump(pipeline, 'nystroem_svm_model.pkl')
    # Fit and evaluate the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_valid)
    y_pred_proba = pipeline.predict_proba(X_valid)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': roc_auc_score(y_valid, y_pred_proba)
    }
    print("Nystroem + Linear SVM Metrics:")
    return metrics
######################################################################

def sgd_svm_model(X_train, y_train, X_valid, y_valid):
    print("################### sgd svm ###############")
    sgd_svm = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-5, penalty='elasticnet',l1_ratio=0.2,alpha=0.0001 ,random_state=42)

    # Fit and evaluate the model
    sgd_svm.fit(X_train, y_train)
    joblib.dump(sgd_svm, 'models/sgd_svm_model.pkl')
    y_pred = sgd_svm.predict(X_valid)

    # Note: SGDClassifier does not directly support probability prediction, so we skip ROC-AUC
    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': 'Not available for SGDClassifier'
    }
    print("SGDClassifier Linear SVM Metrics:")
    return metrics

def xgboost_model(X_train, y_train, X_valid, y_valid, optimize=False):
    print("################### XGBoost ###############")
    
    # Default best parameters if grid search is not used
    default_params = {
        'n_estimators': 300 ,
        'max_depth': 20 ,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0
        }
    
    # Initialize model with or without grid search
    if optimize:
        param_grid = {
            'n_estimators': [300, 500],
            'max_depth': [10, 15, 20],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        grid_search = GridSearchCV(
            xgb.XGBClassifier(tree_method='hist', device='cuda', use_label_encoder=False),
            param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"XGBoost - Best Params from GridSearch: {grid_search.best_params_}")
    else:
        # Use default best params
        best_model = xgb.XGBClassifier(
            n_estimators=default_params['n_estimators'],
            max_depth=default_params['max_depth'],
            learning_rate=default_params['learning_rate'],
            subsample=default_params['subsample'],
            colsample_bytree=default_params['colsample_bytree'],
            # reg_alpha=default_params['reg_alpha'],
            # reg_lambda=default_params['reg_lambda'],
            tree_method='hist',
            device='cuda',
            use_label_encoder=False
        )
        best_model.fit(X_train, y_train)
        print(f"XGBoost - Using Default Params: {default_params}")
    
    # Save the model
    joblib.dump(best_model, 'models/XGboost_model.pkl')

    # Predict using the best model
    y_pred = best_model.predict(X_valid)
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': roc_auc_score(y_valid, y_pred_proba)
    }

    print("XGBoost Metrics:", metrics)
    return metrics

def lightgbm_model(X_train, y_train, X_valid, y_valid, optimize=False):
    print("################### LightGBM ###############")
    
    # Default best parameters if grid search is not used
    default_params = {'n_estimators': 700 , 'num_leaves': 93, 'learning_rate': 0.1}
    
    # Initialize model with or without grid search
    if optimize:
        param_grid = {
            'n_estimators': [100, 250],
            'num_leaves': [31, 62],
            'learning_rate': [0.01, 0.1]
        }
        grid_search = GridSearchCV(
            lgb.LGBMClassifier(device='gpu', boosting_type='gbdt'),
            param_grid,
            cv=StratifiedKFold(n_splits=3),
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"LightGBM - Best Params from GridSearch: {grid_search.best_params_}")
    else:
        # Use default best params
        best_model = lgb.LGBMClassifier(
            n_estimators=default_params['n_estimators'],
            num_leaves=default_params['num_leaves'],
            learning_rate=default_params['learning_rate'],
            device='gpu',
            boosting_type='gbdt'
            #boosting_type='dart'
        )
        best_model.fit(X_train, y_train)
        print(f"LightGBM - Using Default Params: {default_params}")
    
    # Make predictions
    y_pred = best_model.predict(X_valid)
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': roc_auc_score(y_valid, y_pred_proba)
    }

    # Save the model
    joblib.dump(best_model, 'models/lightgbm_model.pkl')
    
    print("LightGBM Metrics:", metrics)
    return metrics

def catboost_model(X_train, y_train, X_valid, y_valid):
    print("################### Catboost ###############")
    # Initialize CatBoost with specific parameters
    model = CatBoostClassifier(
        task_type='GPU',
        iterations=5000,                # Increase iterations if memory allows
        depth=16,                       # Increase depth to capture complexity
        learning_rate=0.1,            # Lower learning rate for finer adjustments
        l2_leaf_reg=10,                 # Slightly increase regularization
        early_stopping_rounds=100,
        verbose=100
    )


    # Train the model
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

    # Save the model
    joblib.dump(model, 'models/catboost_model.pkl')

    # Make predictions
    y_pred = model.predict(X_valid)
    y_pred_proba = model.predict_proba(X_valid)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': roc_auc_score(y_valid, y_pred_proba)
    }
    print("CatBoost Metrics:", metrics)
    return metrics

def stacked_model(X_train, y_train, X_valid, y_valid):
    print("################### Improved Stacked Model with Cross-Validation ###############")
    
    # Ensure data is in numpy array format for consistent indexing
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)

    # Initialize base models with optimized parameters
    rf = RandomForestClassifier(n_estimators=500, max_depth=25, min_samples_split=5, max_features='sqrt', random_state=42, n_jobs=-1, bootstrap=True)
    xgb = XGBClassifier(n_estimators=500, max_depth=20, learning_rate=0.1, tree_method='hist', subsample=0.9, colsample_bytree=0.9)
    lgb = LGBMClassifier(n_estimators=500, num_leaves=62, learning_rate=0.1, boosting_type='gbdt')
    # lgb = LGBMClassifier(n_estimators=500, num_leaves=62, learning_rate=0.1, min_data_in_leaf=20, boosting_type='gbdt')
    # cat = CatBoostClassifier(iterations=500, depth=16, learning_rate=0.1, l2_leaf_reg=10, verbose=100, task_type="GPU", early_stopping_rounds=200,random_state=42)

    # Cross-validation for out-of-fold predictions
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True, random_state=42)
    
    # Placeholder arrays for stacked model features
    stacked_train = np.zeros((X_train.shape[0], 4))  # Placeholder for stacked training features
    stacked_valid = np.zeros((X_valid.shape[0], 4))  # Placeholder for stacked validation features

    for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Fitting fold {i + 1}/{n_splits}...")
        
        # Split data for the current fold
        X_fold_train, X_fold_valid = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_valid = y_train[train_idx], y_train[val_idx]
        
        # Fit models on the fold
        rf.fit(X_fold_train, y_fold_train)
        xgb.fit(X_fold_train, y_fold_train)
        lgb.fit(X_fold_train, y_fold_train)
        # cat.fit(X_fold_train, y_fold_train)
        
        # Generate out-of-fold predictions for stacking
        stacked_train[val_idx, 0] = rf.predict_proba(X_fold_valid)[:, 1]
        stacked_train[val_idx, 1] = xgb.predict_proba(X_fold_valid)[:, 1]
        stacked_train[val_idx, 2] = lgb.predict_proba(X_fold_valid)[:, 1]
        # stacked_train[val_idx, 3] = cat.predict_proba(X_fold_valid)[:, 1]

    # Generate predictions on the validation set for each base model
    stacked_valid[:, 0] = rf.predict_proba(X_valid)[:, 1]
    stacked_valid[:, 1] = xgb.predict_proba(X_valid)[:, 1]
    stacked_valid[:, 2] = lgb.predict_proba(X_valid)[:, 1]
    # stacked_valid[:, 3] = cat.predict_proba(X_valid)[:, 1]

    # Meta-learner: Gradient Boosting Classifier
    print("Meta Learner start")
    meta_learner = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=20, min_samples_split=5, random_state=42)
    meta_learner.fit(stacked_train, y_train)
    
    # Save the meta-learner
    joblib.dump(meta_learner, 'models/enhanced_stacked_meta_model_v2.pkl')

    # Final prediction by meta-learner
    y_pred = meta_learner.predict(stacked_valid)
    y_pred_proba = meta_learner.predict_proba(stacked_valid)[:, 1]

    # Evaluate performance
    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': roc_auc_score(y_valid, y_pred_proba)
    }
    print("Enhanced Stacked Model Metrics:", metrics)
    
    return metrics

def bagging_model(X_train, y_train, X_valid, y_valid, optimize=False):
    print("################### Bagging Classifier ###############")
    
    # Default best parameters if grid search is not used
    default_params = {'n_estimators': 220, 'max_samples': 0.5, 'max_features': 1.0}
    
    # Initialize model with or without grid search
    if optimize:
        param_grid = {
            'n_estimators': [220, 300, 500],
            'max_samples': [0.5, 1.0],
            'max_features': [0.5, 1.0, 'sqrt']
        }
        grid_search = GridSearchCV(
            BaggingClassifier(random_state=42),
            param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"BaggingClassifier - Best Params from GridSearch: {grid_search.best_params_}")
    else:
        # Use default best params
        best_model = BaggingClassifier(
            n_estimators=default_params['n_estimators'],
            max_samples=default_params['max_samples'],
            max_features=default_params['max_features'],
            random_state=42
        )
        best_model.fit(X_train, y_train)
        print(f"BaggingClassifier - Using Default Params: {default_params}")
    
    # Make predictions
    y_pred = best_model.predict(X_valid)
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': roc_auc_score(y_valid, y_pred_proba)
    }

    # Save the model
    joblib.dump(best_model, 'models/bagging_model.pkl')
    
    print("Bagging Classifier Metrics:", metrics)
    return metrics

def adaboost_model(X_train, y_train, X_valid, y_valid, optimize=False):
    print("################### AdaBoost ###############")
    
    # Default best parameters if grid search is not used
    default_params = {'n_estimators': 500, 'learning_rate': 0.1 , 'estimator': DecisionTreeClassifier(max_depth=2)}
    
    # Initialize model with or without grid search
    if optimize:
        param_grid = {
            'n_estimators': [200, 250, 300],
            'learning_rate': [0.05, 0.1],
            'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)]
        }
        grid_search = GridSearchCV(
            AdaBoostClassifier(random_state=42),
            param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        print(f"AdaBoostClassifier - Best Params from GridSearch: {grid_search.best_params_}")
    else:
        # Use default best params
        best_model = AdaBoostClassifier(
            n_estimators=default_params['n_estimators'],
            learning_rate=default_params['learning_rate'],
            random_state=42
        )
        best_model.fit(X_train, y_train)
        print(f"AdaBoostClassifier - Using Default Params: {default_params}")
    
    # Make predictions
    y_pred = best_model.predict(X_valid)
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]

    # Calculate evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': roc_auc_score(y_valid, y_pred_proba)
    }

    # Save the model
    joblib.dump(best_model, 'models/adaboost_model.pkl')
    
    print("AdaBoost Metrics:", metrics)
    return metrics



def voting_ensemble(X_train, y_train, X_valid, y_valid):
    print("################### Voting Ensemble ###############")
    # Define base models
    base_models = [
        ('xgb', xgb.XGBClassifier(tree_method='hist', device='cuda' )),
        ('lgbm', lgb.LGBMClassifier(device='gpu', boosting_type='gbdt',n_estimators= 1500 ,num_leaves=93,learning_rate=0.1)),
    ]
    
    # Create voting classifier
    voting_model = VotingClassifier(estimators=base_models, voting='soft')
    
    # Train the model
    voting_model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(voting_model, 'models/voting_model.pkl')
    
    # Predict using the voting model
    y_pred = voting_model.predict(X_valid)
    y_pred_proba = voting_model.predict_proba(X_valid)[:, 1]
    
    # Calculate evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_valid, y_pred),
        'Precision': precision_score(y_valid, y_pred),
        'Recall': recall_score(y_valid, y_pred),
        'F1-score': f1_score(y_valid, y_pred),
        'ROC-AUC': roc_auc_score(y_valid, y_pred_proba)
    }
    
    print("Voting Ensemble Metrics:", metrics)
    return metrics


# Define the function to execute the models and measure execution time
def execute_models(X_train_split,Y_train_split,X_valid,y_valid):
    import time 
    start_time = time.time()
    
    results = {}
    execution_times = {}

    start_time = time.time()
    results['Logistic Regression'] = logistic_regression_model(X_train_split, Y_train_split, X_valid, y_valid)
    execution_times['Logistic Regression'] = time.time() - start_time

    start_time = time.time()
    results['Decision Tree'] = decision_tree_model(X_train_split, Y_train_split, X_valid, y_valid)
    execution_times['Decision Tree'] = time.time() - start_time

    start_time = time.time()
    results['Random Forest'] = random_forest_model(X_train_split, Y_train_split, X_valid, y_valid)
    execution_times['Random Forest'] = time.time() - start_time

    start_time = time.time()
    results['Bagging Model'] = bagging_model(X_train_split, Y_train_split, X_valid, y_valid)
    execution_times['Bagging Model'] = time.time() - start_time

    start_time = time.time()
    results['Adaboost_model'] = adaboost_model(X_train_split, Y_train_split, X_valid, y_valid)
    execution_times['Adaboost_model'] = time.time() - start_time

    start_time = time.time()
    results['SGDClassifier Linear SVM'] = sgd_svm_model(X_train_split, Y_train_split, X_valid, y_valid)
    execution_times['SGDClassifier Linear SVM'] = time.time() - start_time

    start_time = time.time()
    results['XGBoost'] = xgboost_model(X_train_split, Y_train_split, X_valid, y_valid)
    execution_times['XGBoost'] = time.time() - start_time

    start_time = time.time()
    results['LightGBM'] = lightgbm_model(X_train_split, Y_train_split, X_valid, y_valid)
    execution_times['LightGBM'] = time.time() - start_time

    start_time = time.time()
    results['CatBoost'] = catboost_model(X_train_split, Y_train_split, X_valid, y_valid)
    execution_times['CatBoost'] = time.time() - start_time

    start_time = time.time()
    results['stacked_model'] = stacked_model(X_train_split, Y_train_split, X_valid, y_valid)
    execution_times['stacked_model'] = time.time() - start_time

    start_time = time.time()
    results['voting_ensemble'] = voting_ensemble(X_train_split, Y_train_split, X_valid, y_valid)
    execution_times['voting_ensemble'] = time.time() - start_time

    # Display comparison of models
    results_df = pd.DataFrame(results).T
    print("\nComparison of Model Performance:")
    print(results_df)

    # Display execution times
    execution_times_df = pd.DataFrame.from_dict(execution_times, orient='index', columns=['Execution Time (seconds)'])
    print("\nExecution Time of Each Model:")
    print(execution_times_df)

def save_model_results(ids, X_valid, filename='Submission/sourena_mohit_submission.csv',model_name='voting_model.pkl'):
    model = joblib.load(model_name)
    y_pred = model.predict(X_valid)
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'ID': ids,
        'Overall_Experience': y_pred
    })
    
    # Save the DataFrame to a CSV file
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def modeling():
    train,test = load()
    train,test = ID_Change(train,test)
    train,test = float_to_int_for_memory_management(train,test)
    train,test = Interaction_Feature(train,test)
    print("#################################################")
    train,test = combining_satisfaction_features(train,test)
    print("#################################################")
    # Check class distribution
    print("Class distribution in Overall_Experience:")
    print(train["Overall_Experience"].value_counts(normalize=True))
    """
    Given the distribution of your target variable (Overall_Experience), with approximately 54.7% for class 1 and 45.3% for class 0, this is not a highly imbalanced dataset. Typically, 
    SMOTE and other oversampling techniques are used when one class significantly outweighs the other, such as an 80-20 split or greater imbalance.Recommendation:
    SMOTE is not necessary for your current class distribution because the classes are relatively balanced.
    """
    train, test = encode_categorical_features(train, test)
    print(train.info())
    print(test.info())

    # seperating the independant and dependant variables
    ID_train = train['ID']
    X_train = train.drop(['ID', 'Overall_Experience', 'Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins'], axis = 1)
    Y_train = train['Overall_Experience']
    ID_test = test['ID']
    X_test = test.drop(['ID', 'Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins'], axis = 1)
    # scale the data if needed
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)        

    print("#################################################")
    print("##### MODELING #####")
    print("#################################################")
    from sklearn.model_selection import train_test_split
    # performing train-test split of (80:20) on the training data
    X_train_split, X_valid, Y_train_split, y_valid = train_test_split(X_train_scaled, Y_train, test_size=0.2, random_state=42)
    print(X_train_split.shape)
    print(X_valid.shape)
    print(Y_train_split.shape)
    print(y_valid.shape)
    execute_models(X_train_split,Y_train_split,X_valid,y_valid)
    print("modeling is Done")
    ###################################### I did the modeling part ###################################################
    """
    Analysis of Model Performance
    Your models show a range of performance metrics, with particular strengths in accuracy, F1-score, and ROC-AUC for ensemble methods like Random Forest, XGBoost, LightGBM, and CatBoost. Here’s a summary analysis:

    Top Performers in Accuracy, F1, and ROC-AUC:

    XGBoost and LightGBM achieved the highest accuracy, F1-scores, and ROC-AUC values, indicating they are consistently strong in predictive power and handling class balance.
    Random Forest also performs well across all metrics and has the advantage of interpretability compared to other ensemble methods.
    CatBoost has slightly lower performance than XGBoost and LightGBM but remains competitive, especially if you have categorical data, as it can handle this directly.
    Lower Performing Models:

    Logistic Regression and SGDClassifier (Linear SVM): Both have comparatively lower accuracy and F1-scores and should likely be dropped. They are simple and fast but may not capture the complex patterns in the data as well as ensemble models.
    Adaboost: While better than Logistic Regression and Linear SVM, it still lags behind XGBoost and LightGBM in terms of all metrics, particularly ROC-AUC.
    Execution Time Considerations:

    CatBoost and the stacked model have very high execution times, which may be prohibitive for practical use if efficiency is important.
    Random Forest and Bagging Model also have moderate execution times but are faster than CatBoost and the stacked model.
    XGBoost and LightGBM provide a balanced combination of high performance and reasonable execution time, making them efficient and effective choices.
    """
    ######################################################################################################################
    print('voting_ensemble: chosen model')
    print('Based on the metrics, the Voting Ensemble method is the best choice.\n It combines the strengths of XGBoost and LightGBM, resulting in the highest overall performance across most metrics.')
    print('saving the result of test as final submission')
    save_model_results(ids=ID_test,X_valid=X_test_scaled,model_name='voting_model.pkl',filename='Submission/sourena_mohit_Voting_Model_result.csv')
    save_model_results(ids=ID_test,X_valid=X_test_scaled,model_name='XGboost_model.pkl',filename='Submission/sourena_mohit_XGboost_result.csv')
    save_model_results(ids=ID_test,X_valid=X_test_scaled,model_name='lightgbm_model.pkl',filename='Submission/sourena_mohit_lightgbm_result.csv')


    



   