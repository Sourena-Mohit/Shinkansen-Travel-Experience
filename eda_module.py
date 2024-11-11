# EDA.py
from init import pd , np , os , plt , sns , skew , output_dir , bivariate_dir

# Section 1 to 6 (previously implemented functions) would be here...
# Section 1: Load Data
def load_data():
    global travel_data_train, survey_data_train, travel_data_test, survey_data_test
    travel_data_train = pd.read_csv('Data/Train/Traveldata_train.csv')
    survey_data_train = pd.read_csv('Data/Train/Surveydata_train.csv')
    travel_data_test = pd.read_csv('Data/Test/Traveldata_test.csv')
    survey_data_test = pd.read_csv('Data/Test/Surveydata_test.csv')
    
    print("Training Travel Data:")
    print(travel_data_train.head())
    print("Training Survey Data:")
    print(survey_data_train.head())
    print("Test Travel Data:")
    print(travel_data_test.head())
    print("Test Survey Data:")
    print(survey_data_test.head())

    return travel_data_train, survey_data_train, travel_data_test, survey_data_test

# Section 2: Data Overview
def data_overview():
    datasets = {
        "Training Travel Data": travel_data_train,
        "Training Survey Data": survey_data_train,
        "Test Travel Data": travel_data_test,
        "Test Survey Data": survey_data_test,
    }
    for name, data in datasets.items():
        print(f"\n{name} Info:")
        print(data.info())
        print(f"{name} - Summary Statistics:")
        print(data.describe())

# Section 3: Missing Values and Duplicates
def check_missing_and_duplicates():
    datasets = {
        "Training Travel Data": travel_data_train,
        "Training Survey Data": survey_data_train,
        "Test Travel Data": travel_data_test,
        "Test Survey Data": survey_data_test,
    }
    for name, data in datasets.items():
        print(f"\nMissing Values - {name}:")
        print(data.isnull().sum())
        print(f"Duplicate Rows in {name}:", data.duplicated().sum())

# Section 4: Distribution of Categorical Variables
def categorical_distributions():
    datasets = {
        "Training Travel Data": travel_data_train,
        "Training Survey Data": survey_data_train,
        "Test Travel Data": travel_data_test,
        "Test Survey Data": survey_data_test,
    }
    for name, data in datasets.items():
        print(f"\nCategorical Distributions in {name}:")
        for col in data.select_dtypes(include='object').columns:
            print(f"{col} Distribution:")
            print(data[col].value_counts())

# Section 5: Checking Extreme Values
def check_extreme_values():
    datasets = {
        "Training Travel Data": travel_data_train,
        "Test Travel Data": travel_data_test,
    }
    columns_to_check = ['Age', 'Travel_Distance', 'Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins']
    
    for name, data in datasets.items():
        print(f"\n{name} - Checking Extreme Values")
        for col in columns_to_check:
            if col in data.columns:
                print(f"Summary Statistics for {col} in {name}:")
                print(data[col].describe())

# Section 6: Cap Extreme Values at 5th and 95th Percentiles
def cap_extreme_values():
    datasets = {
        "Training Travel Data": travel_data_train,
        "Test Travel Data": travel_data_test,
    }
    columns_to_check = ['Age', 'Travel_Distance', 'Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins']
    
    for name, data in datasets.items():
        percentiles = {col: data[col].quantile([0.05, 0.95]) for col in columns_to_check if col in data.columns}
        for col in columns_to_check:
            if col in data.columns:
                lower_cap, upper_cap = percentiles[col][0.05], percentiles[col][0.95]
                data[col] = np.where(data[col] < lower_cap, lower_cap, data[col])
                data[col] = np.where(data[col] > upper_cap, upper_cap, data[col])
        print(f"Capped extreme values in {name} at the 5th and 95th percentiles.")


# Section 7: Univariate Analysis for Numerical Variables
# eda_module.py

def univariate_analysis_numerical(df, column, dataset_type="train"):
    """
    Analyze and save univariate analysis plots for a numerical column.
    """
    # Check if the column exists in the dataframe
    if column not in df.columns:
        print(f"Column '{column}' not found in {dataset_type} dataset. Skipping analysis.")
        return
    
    print(f"Descriptive Statistics for {column} in {dataset_type} dataset:")
    print(df[column].describe())
    print("\n")
    
    plt.figure(figsize=(14, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f"Histogram of {column} - {dataset_type.capitalize()}")
    
    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column} - {dataset_type.capitalize()}")
    
    # Save the plot as an image with dataset type in filename
    plt.savefig(os.path.join(output_dir, f"{dataset_type}_{column}_univariate_analysis.png"))
    plt.close()


def univariate_analysis_categorical(df, column, dataset_type="train"):
    """
    Analyze and save univariate analysis plots for a categorical column.
    """
    # Check if the column exists in the dataframe
    if column not in df.columns:
        print(f"Column '{column}' not found in {dataset_type} dataset. Skipping analysis.")
        return
    
    print(f"Value Counts for {column} in {dataset_type} dataset:")
    print(df[column].value_counts(dropna=False))
    print("\n")
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column, order=df[column].value_counts().index)
    plt.title(f"Distribution of {column} - {dataset_type.capitalize()}")
    plt.xticks(rotation=45)
    
    # Save the plot as an image with dataset type in filename
    plt.savefig(os.path.join(output_dir, f"{dataset_type}_{column}_distribution.png"))
    plt.close()


# eda_module.py

# Perform univariate analysis for train and test datasets
def perform_univariate_analysis(train_dfs, test_dfs):
    # Unpack datasets
    travel_data_train, survey_data_train = train_dfs
    travel_data_test, survey_data_test = test_dfs

    # Numerical columns in Travel Data
    print("Travel Data - Numerical Variables Analysis\n")
    numerical_columns = ['Age', 'Travel_Distance', 'Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins']
    for column in numerical_columns:
        univariate_analysis_numerical(travel_data_train, column, dataset_type="train")
        univariate_analysis_numerical(travel_data_test, column, dataset_type="test")
    
    # Numerical column in Survey Data
    print("Survey Data - Numerical Variable Analysis\n")
    univariate_analysis_numerical(survey_data_train, 'Overall_Experience', dataset_type="train")
    univariate_analysis_numerical(survey_data_test, 'Overall_Experience', dataset_type="test")

    # Categorical columns in Travel Data    
    print("Travel Data - Categorical Variables Analysis\n")
    categorical_columns_travel = ['Gender', 'Customer_Type', 'Type_Travel', 'Travel_Class']
    for column in categorical_columns_travel:
        univariate_analysis_categorical(travel_data_train, column, dataset_type="train")
        univariate_analysis_categorical(travel_data_test, column, dataset_type="test")

    # Categorical columns in Survey Data
    print("Survey Data - Categorical Variables Analysis\n")
    categorical_columns_survey = [
        'Seat_Comfort', 'Seat_Class', 'Arrival_Time_Convenient', 'Catering', 'Platform_Location',
        'Onboard_Wifi_Service', 'Onboard_Entertainment', 'Online_Support', 'Ease_of_Online_Booking',
        'Onboard_Service', 'Legroom', 'Baggage_Handling', 'CheckIn_Service', 'Cleanliness', 'Online_Boarding'
    ]
    for column in categorical_columns_survey:
        univariate_analysis_categorical(survey_data_train, column, dataset_type="train")
        univariate_analysis_categorical(survey_data_test, column, dataset_type="test")


# Section 9: Function to calculate and print skewness for numerical columns
def analyze_skewness(df, numerical_columns, threshold_moderate=0.5, threshold_high=1.0):
    """
    Calculate and print skewness for each numerical column in the given DataFrame.

    Parameters:
    - df (DataFrame): The dataframe containing the data.
    - numerical_columns (list): List of numerical column names to check skewness.
    - threshold_moderate (float): Threshold above which skewness is considered moderate.
    - threshold_high (float): Threshold above which skewness is considered high.

    Returns:
    - skewness_results (dict): Dictionary with column names as keys and skewness values.
    """
    skewness_results = {}
    
    print("\nSkewness Analysis\n")
    for column in numerical_columns:
        skewness = skew(df[column].dropna())  # Drop NaN values to calculate skewness
        skewness_results[column] = skewness
        
        print(f"Skewness of {column}: {skewness:.2f}")
        if abs(skewness) > threshold_high:
            print(f"   - {column} is highly skewed. Consider log or Box-Cox transformation.\n")
        elif abs(skewness) > threshold_moderate:
            print(f"   - {column} is moderately skewed. Consider square root transformation.\n")
        else:
            print(f"   - {column} is approximately symmetric. No transformation needed.\n")
    
    return skewness_results

#Section 10
def apply_log_transformation(df, columns):
    """
    Applies log transformation to specified columns to reduce skewness.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - columns (list): List of column names to apply log transformation.

    Returns:
    - df (DataFrame): Updated DataFrame with log-transformed columns.
    """
    # Apply log transformation to each specified column
    for column in columns:
        log_column = f"Log_{column}"
        df[log_column] = np.log1p(df[column])  # np.log1p to handle zero values
        
        # Check skewness after transformation
        transformed_skewness = df[log_column].skew()
        print(f"Skewness of {log_column}: {transformed_skewness:.2f}")
        
        # Report on transformation effect
        if abs(transformed_skewness) < 0.5:
            print(f"   - {log_column} is now approximately symmetric.\n")
        else:
            print(f"   - {log_column} is still skewed, consider further transformation if needed.\n")

    return df

# Section 11: Merging and Checking Datasets
def merge_and_save_datasets(travel_df, survey_df, on_column='ID', how='inner', dataset_type="train"):
    """
    Merges the travel and survey datasets on a specified column, saves to a file, and performs basic checks.
    
    Parameters:
    - travel_df (DataFrame): Travel data.
    - survey_df (DataFrame): Survey data.
    - on_column (str): Column to merge on (default is 'ID').
    - how (str): Type of merge (default is 'inner').
    - dataset_type (str): Either 'train' or 'test' to distinguish datasets.
    
    Returns:
    - combined_data (DataFrame): Merged DataFrame with basic checks applied.
    """
    # Merge datasets
    combined_data = travel_df.merge(survey_df, on=on_column, how=how)
    print(f"{dataset_type.capitalize()} Combined Data - Shape: {combined_data.shape}")
    
    # Check for duplicate IDs and missing values
    duplicates = combined_data.duplicated(subset=on_column).sum()
    if duplicates > 0:
        print(f"Warning: {duplicates} duplicate IDs found in {dataset_type} combined data.")
    
    missing_values = combined_data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        print(f"Missing values in {dataset_type} combined data:")
        print(missing_values)
    
    # Resolve column conflicts
    combined_data.columns = [col.replace('_x', '_Travel').replace('_y', '_Survey') for col in combined_data.columns]
    
    # Save the combined dataset
    save_path = f"Data/{dataset_type}_combined_data.csv"
    combined_data.to_csv(save_path, index=False)
    print(f"Saved {dataset_type} combined data to {save_path}")
    
    return combined_data

def simple_analysis_summary(combined_data):
    """
    Performs a simple analysis on the combined dataset.
    
    Parameters:
    - combined_data (DataFrame): The merged DataFrame to analyze.
    
    Returns:
    - None: Prints out basic statistics and information.
    """
    print("\n--- Simple Analysis Summary ---")
    print("Shape of Combined Data:", combined_data.shape)
    print("Column Data Types:\n", combined_data.dtypes)
    print("\nSummary Statistics:\n", combined_data.describe(include='all'))
    print("\nMissing Values:\n", combined_data.isnull().sum()[combined_data.isnull().sum() > 0])

# Function to merge both train and test datasets and save them
def process_and_save_all_datasets(travel_data_train, survey_data_train, travel_data_test, survey_data_test):
    # Merge and save train data
    combined_train = merge_and_save_datasets(travel_data_train, survey_data_train, dataset_type="train")
    
    # Merge and save test data
    combined_test = merge_and_save_datasets(travel_data_test, survey_data_test, dataset_type="test")
    
    # Perform simple analysis on the merged datasets
    print("\n--- Analysis for Train Combined Data ---")
    simple_analysis_summary(combined_train)
    
    print("\n--- Analysis for Test Combined Data ---")
    simple_analysis_summary(combined_test)


# Section 12: Bivariate Analysis

# 1. Numerical vs. Numerical Analysis (Correlation Heatmap and Scatter Plots)
def numerical_vs_numerical_analysis(df, numerical_columns, dataset_type="train"):
    """
    Visualize correlation and scatter plots between numerical columns.

    Parameters:
    - df: DataFrame containing the data
    - numerical_columns: List of numerical columns for the analysis
    - dataset_type (str): 'train' or 'test' to distinguish dataset type in file names
    """
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df[numerical_columns].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"{dataset_type.capitalize()} Correlation Heatmap - Numerical Variables")
    plt.savefig(os.path.join(bivariate_dir, f"{dataset_type}_Numerical_Correlation_Heatmap.png"))
    plt.close()

    # Pairplot for scatter plots between pairs of numerical columns
    pairplot = sns.pairplot(df[numerical_columns], diag_kind="kde")
    pairplot.fig.suptitle(f"{dataset_type.capitalize()} Scatter Plot Matrix - Numerical Variables", y=1.02)
    pairplot.savefig(os.path.join(bivariate_dir, f"{dataset_type}_Numerical_Scatter_Matrix.png"))
    plt.close()

# 2. Categorical vs. Numerical Analysis (Boxplots)
def categorical_vs_numerical_analysis(df, categorical_column, numerical_column, dataset_type="train"):
    """
    Visualize the relationship between a categorical and a numerical column using boxplots.

    Parameters:
    - df: DataFrame containing the data
    - categorical_column: The categorical column for grouping
    - numerical_column: The numerical column to analyze
    - dataset_type (str): 'train' or 'test' to distinguish dataset type in file names
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=categorical_column, y=numerical_column)
    plt.title(f"{dataset_type.capitalize()} Boxplot of {numerical_column} by {categorical_column}")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(bivariate_dir, f"{dataset_type}_Boxplot_{numerical_column}_by_{categorical_column}.png"))
    plt.close()

# 3. Categorical vs. Categorical Analysis (Count Plots and Cross-tabulations)
def categorical_vs_categorical_analysis(df, column1, column2, dataset_type="train"):
    """
    Visualize the relationship between two categorical columns using count plots and cross-tabulation.

    Parameters:
    - df: DataFrame containing the data
    - column1: The first categorical column
    - column2: The second categorical column for the hue
    - dataset_type (str): 'train' or 'test' to distinguish dataset type in file names
    """
    # Count plot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column1, hue=column2)
    plt.title(f"{dataset_type.capitalize()} Count Plot of {column1} by {column2}")
    plt.xticks(rotation=45)
    plt.legend(title=column2)
    plt.savefig(os.path.join(bivariate_dir, f"{dataset_type}_CountPlot_{column1}_by_{column2}.png"))
    plt.close()

    # Cross-tabulation
    crosstab = pd.crosstab(df[column1], df[column2], normalize='index')
    print(f"Cross-tabulation of {column1} and {column2} in {dataset_type} data:")
    print(crosstab)
    print("\n")