# main.py
from init import pd
from eda_module import (
    load_data,
    data_overview,
    check_missing_and_duplicates,
    categorical_distributions,
    check_extreme_values,
    cap_extreme_values,
    perform_univariate_analysis,
    analyze_skewness,
    apply_log_transformation,
    process_and_save_all_datasets,
    numerical_vs_numerical_analysis,
    categorical_vs_categorical_analysis,
    categorical_vs_numerical_analysis
)

from data_pre import data_pre_processing
from Modeling import modeling

def eda():
    print("Starting EDA Process...")
    
    # Step 1: Load Data
    print("\n--- Loading Data ---")
    travel_data_train, survey_data_train, travel_data_test, survey_data_test = load_data()
    
    # Step 2: Data Overview
    print("\n--- Data Overview ---")
    data_overview()
    
    # Step 3: Check Missing Values and Duplicates
    print("\n--- Checking Missing Values and Duplicates ---")
    check_missing_and_duplicates()
    
    # Step 4: Distribution of Categorical Variables
    print("\n--- Analyzing Categorical Distributions ---")
    categorical_distributions()
    
    # Step 5: Checking Extreme Values
    print("\n--- Checking Extreme Values ---")
    check_extreme_values()
    
    # Step 6: Cap Extreme Values at 5th and 95th Percentiles
    print("\n--- Capping Extreme Values ---")
    cap_extreme_values()
    
    # Step 7 to 8: Univariate Analysis for Numerical and Categorical Variables
    print("\n--- Performing Univariate Analysis ---")
    perform_univariate_analysis(
        (travel_data_train, survey_data_train),
        (travel_data_test, survey_data_test)
    )

    # Step 9: Skewness Analysis for Train and Test Data
    print("\n--- Skewness Analysis for Train Data ---")
    numerical_columns_train = ['Age', 'Travel_Distance', 'Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins']
    analyze_skewness(travel_data_train, numerical_columns_train)
    
    print("\n--- Skewness Analysis for Test Data ---")
    analyze_skewness(travel_data_test, numerical_columns_train)  # Same columns for test data

    # Step 10: Log Transformation
    columns_to_transform = ['Age', 'Travel_Distance', 'Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins']
    print("\n--- Applying Log Transformation on Train Data ---")
    travel_data_train = apply_log_transformation(travel_data_train, columns_to_transform)
    
    print("\n--- Applying Log Transformation on Test Data ---")
    travel_data_test = apply_log_transformation(travel_data_test, columns_to_transform)
    
    # Step 11: Bivariate Analysis for Train and Test Data

    # Numerical vs Numerical
    print("\n--- Bivariate Analysis: Numerical vs Numerical ---")
    numerical_columns = ['Age', 'Travel_Distance', 'Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins']
    numerical_vs_numerical_analysis(travel_data_train, numerical_columns, dataset_type="train")
    numerical_vs_numerical_analysis(travel_data_test, numerical_columns, dataset_type="test")
    
    # Categorical vs Numerical
    print("\n--- Bivariate Analysis: Categorical vs Numerical ---")
    categorical_columns = ['Gender', 'Customer_Type', 'Type_Travel', 'Travel_Class']
    for cat_col in categorical_columns:
        for num_col in numerical_columns:
            categorical_vs_numerical_analysis(travel_data_train, cat_col, num_col, dataset_type="train")
            categorical_vs_numerical_analysis(travel_data_test, cat_col, num_col, dataset_type="test")
    
    # Categorical vs Categorical
    print("\n--- Bivariate Analysis: Categorical vs Categorical ---")
    categorical_pairs = [
        ('Gender', 'Customer_Type'),
        ('Type_Travel', 'Travel_Class')
    ]
    for col1, col2 in categorical_pairs:
        categorical_vs_categorical_analysis(travel_data_train, col1, col2, dataset_type="train")
        categorical_vs_categorical_analysis(travel_data_test, col1, col2, dataset_type="test")


    # Step 12: Merge, Save, and Analyze Combined Datasets
    print("\n--- Merging, Saving, and Analyzing Combined Datasets ---")
    process_and_save_all_datasets(travel_data_train, survey_data_train, travel_data_test, survey_data_test)
    
    print("EDA Process Completed")


def main():
    eda()
    data_pre_processing()
    modeling()


if __name__ == "__main__":
    main()

