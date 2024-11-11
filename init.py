##############################################
# Import Libraries
##############################################

# Standard Libraries
import os
import warnings
warnings.filterwarnings('ignore')

# Data Manipulation
import pandas as pd
import numpy as np

# Statistical and Data Preprocessing
from scipy.stats import skew, boxcox, yeojohnson
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from scipy.stats import mode

# Model Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Sampling and Encoding
from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder

# Clustering
from sklearn.cluster import KMeans

# Deep Learning Libraries for GPU Manipulation
# import tensorflow as tf  # TensorFlow

# GPU Acceleration

# Directory for saving figures
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# Set output directory for bivariate analysis figures
bivariate_dir = "figures/bivariate"
os.makedirs(bivariate_dir, exist_ok=True)

# Set up visual styling for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)




