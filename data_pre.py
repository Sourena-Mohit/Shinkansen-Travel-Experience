from init import LinearRegression ,LabelEncoder,KNNImputer,KMeans,mode,np,pd

def impute_and_bin_age(df, strategy='median'):
    """
    Impute missing values in the Age column and create an Age_Group column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with the Age column.
    - strategy (str): The imputation strategy ('median', 'mean', 'mode').

    Returns:
    - pd.DataFrame: DataFrame with imputed Age and new Age_Group column.
    """
    # Impute Age
    if strategy == 'median':
        df['Age'].fillna(df['Age'].median(), inplace=True)
    elif strategy == 'mean':
        df['Age'].fillna(df['Age'].mean(), inplace=True)
    elif strategy == 'mode':
        df['Age'].fillna(df['Age'].mode()[0], inplace=True)
    else:
        raise ValueError("Strategy must be 'median', 'mean', or 'mode'")

    # Binning for Age_Group
    age_bins = [0, 18, 35, 55, 100]
    age_labels = ['Youth', 'Adult', 'Middle-aged', 'Senior']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

    # Verification
    print("Handled missing data in column: Age")
    print("Missing values in Age after imputation:", df['Age'].isnull().sum())
    print("Age Group Distribution:")
    print(df['Age_Group'].value_counts())

    return df

def impute_gender_based_on_age_group(df):
    """
    Impute missing values in the 'Gender' column based on the distribution within each 'Age_Group'.
    If 'Age_Group' is missing, fallback to the overall mode of 'Gender'.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame with 'Gender' and 'Age_Group' columns.
    
    Returns:
    - pd.DataFrame: DataFrame with 'Gender' imputed for missing values.
    """
    # Calculate Gender distribution by Age Group for imputation
    age_group_gender_dist = (
        df.dropna(subset=['Gender'])
        .groupby('Age_Group')['Gender']
        .value_counts(normalize=True)
        .unstack()
    )
    
    # Overall mode of Gender as fallback
    gender_mode = df['Gender'].mode()[0]

    # Impute missing 'Gender' values based on Age_Group distribution or fallback to mode
    def impute_gender(row):
        if pd.isna(row['Gender']):
            if not pd.isna(row['Age_Group']):
                # Use Age_Group-based probabilities to assign Gender
                gender_dist = age_group_gender_dist.loc[row['Age_Group']]
                return np.random.choice(gender_dist.index, p=gender_dist.values)
            else:
                # Fallback to overall mode if Age_Group is missing
                return gender_mode
        return row['Gender']

    # Apply imputation
    df['Gender'] = df.apply(impute_gender, axis=1)

    # Verification
    print("Handled missing data in column: Gender")
    print(f"Missing values in Gender after imputation: {df['Gender'].isnull().sum()}")

    return df

def display_missing_data_summary(df, dataset_name):
    """
    Display the columns with remaining missing values in the dataset.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to check for missing values.
    - dataset_name (str): Name of the dataset (train or test) for display purposes.
    """
    missing_data = df.isnull().sum()
    remaining_missing = missing_data[missing_data > 0]
    if remaining_missing.empty:
        print(f"No missing data left in {dataset_name} dataset.")
    else:
        print(f"\nRemaining missing data in {dataset_name} dataset:")
        print(remaining_missing)

def impute_log_delays_with_regression(df):
    """
    Impute missing values in 'Log_Departure_Delay_in_Mins' and 'Log_Arrival_Delay_in_Mins' 
    using linear regression based on their relationship, with mean imputation as a fallback.
    """
    # Identify missing rows for delays
    departure_missing = df['Log_Departure_Delay_in_Mins'].isnull()
    arrival_missing = df['Log_Arrival_Delay_in_Mins'].isnull()

    # Step 1: Predict missing Log_Departure_Delay using Log_Arrival_Delay
    known_data = df[~departure_missing & ~arrival_missing]
    arrival_known = known_data[['Log_Arrival_Delay_in_Mins']]
    departure_known = known_data['Log_Departure_Delay_in_Mins']

    lr_departure = LinearRegression().fit(arrival_known, departure_known)
    arrival_for_prediction = df.loc[departure_missing & ~arrival_missing, ['Log_Arrival_Delay_in_Mins']]
    if not arrival_for_prediction.empty:
        predicted_departure_delay = lr_departure.predict(arrival_for_prediction)
        df.loc[departure_missing & ~arrival_missing, 'Log_Departure_Delay_in_Mins'] = predicted_departure_delay

    # Step 2: Predict missing Log_Arrival_Delay using Log_Departure_Delay
    departure_known = known_data[['Log_Departure_Delay_in_Mins']]
    arrival_known = known_data['Log_Arrival_Delay_in_Mins']

    lr_arrival = LinearRegression().fit(departure_known, arrival_known)
    departure_for_prediction = df.loc[arrival_missing & ~departure_missing, ['Log_Departure_Delay_in_Mins']]
    if not departure_for_prediction.empty:
        predicted_arrival_delay = lr_arrival.predict(departure_for_prediction)
        df.loc[arrival_missing & ~departure_missing, 'Log_Arrival_Delay_in_Mins'] = predicted_arrival_delay

    # Step 3: Fill remaining missing values with the mean
    df['Log_Departure_Delay_in_Mins'].fillna(df['Log_Departure_Delay_in_Mins'].mean(), inplace=True)
    df['Log_Arrival_Delay_in_Mins'].fillna(df['Log_Arrival_Delay_in_Mins'].mean(), inplace=True)

    # Step 4: Reverse log transformation for original delay columns
    df['Departure_Delay_in_Mins'] = np.expm1(df['Log_Departure_Delay_in_Mins'])
    df['Arrival_Delay_in_Mins'] = np.expm1(df['Log_Arrival_Delay_in_Mins'])

    # Verification output
    print("Missing values after delay imputation:")
    print(df[['Departure_Delay_in_Mins', 'Arrival_Delay_in_Mins', 'Log_Departure_Delay_in_Mins', 'Log_Arrival_Delay_in_Mins']].isnull().sum())

    return df

def impute_ordinal_by_group(df, column, group_by_col):
    """
    Impute missing values in an ordinal column based on the mode within each group.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with missing values.
        column (str): The column to impute.
        group_by_col (str): The column to group by for mode-based imputation.
    """
    # Calculate mode for each group and impute missing values
    mode_by_group = df.groupby(group_by_col)[column].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)

    # Apply the imputation
    def apply_mode(row):
        return mode_by_group[row[group_by_col]] if pd.isna(row[column]) else row[column]

    df[column] = df.apply(apply_mode, axis=1)

    # Verification output
    print(f"Missing values in {column} after imputation: {df[column].isnull().sum()}")
    return df
################################################### Extra ##########################################################
def encode_categorical_features(df, columns):
    """
    Encode categorical features using Label Encoding.
    """
    df_encoded = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    return df_encoded, encoders

def decode_categorical_features(df, encoders):
    """
    Decode categorical features using stored Label Encoders.
    """
    for col, le in encoders.items():
        df[col] = le.inverse_transform(df[col].round().astype(int))
    return df

def knn_impute_seat_comfort(df, knn_features=None, n_neighbors=5):
    """
    Impute missing values in 'Seat_Comfort' based on KNN imputation.
    """
    if knn_features is None:
        knn_features = ['Seat_Class', 'Gender', 'Age_Group', 'Cleanliness', 'Travel_Class', 'Seat_Comfort']

    df_encoded, encoders = encode_categorical_features(df, knn_features)
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_encoded[knn_features] = knn_imputer.fit_transform(df_encoded[knn_features])

    df_imputed = decode_categorical_features(df_encoded, {'Seat_Comfort': encoders['Seat_Comfort']})
    print("Missing values after KNN imputation for Seat_Comfort handled.")
    return df_imputed
################################################ Extra #############################################################

def impute_platform_location_and_wifi(df, knn_features=None, n_neighbors=5):
    """
    Impute missing values for 'Platform_Location' and 'Onboard_Wifi_Service' using mode 
    imputation within groups and KNN for remaining missing values.
    """
    df_imputed = df.copy()

    # Step 1: Impute 'Platform_Location' based on 'Travel_Class'
    mode_platform_by_class = df_imputed.groupby('Travel_Class')['Platform_Location'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'Manageable')
    df_imputed['Platform_Location'] = df_imputed.apply(
        lambda row: mode_platform_by_class[row['Travel_Class']] if pd.isna(row['Platform_Location']) else row['Platform_Location'],
        axis=1
    )

    # Step 2: Impute 'Onboard_Wifi_Service' based on 'Onboard_Entertainment'
    mode_wifi_by_entertainment = df_imputed.groupby('Onboard_Entertainment')['Onboard_Wifi_Service'].apply(lambda x: x.mode()[0] if not x.mode().empty else 'Good')

    def impute_wifi(row):
        if pd.isna(row['Onboard_Wifi_Service']):
            if pd.notna(row['Onboard_Entertainment']):
                return mode_wifi_by_entertainment.get(row['Onboard_Entertainment'], 'Good')
            else:
                # Fallback to overall mode if 'Onboard_Entertainment' is missing
                return df_imputed['Onboard_Wifi_Service'].mode()[0]
        return row['Onboard_Wifi_Service']

    df_imputed['Onboard_Wifi_Service'] = df_imputed.apply(impute_wifi, axis=1)

    # Step 3: Apply KNN Imputer for remaining missing values
    if knn_features is None:
        knn_features = ['Platform_Location', 'Onboard_Wifi_Service', 'Travel_Class', 'Seat_Class', 'Onboard_Entertainment']
    
    # Encode categorical columns for KNN compatibility
    df_encoded, label_encoders = encode_categorical_features(df_imputed, knn_features)
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_encoded[knn_features] = knn_imputer.fit_transform(df_encoded[knn_features])
    
    # Decode to restore original categorical values
    df_imputed = decode_categorical_features(df_encoded, label_encoders)
    
    # Verification of imputation
    print("Missing values after imputation for Platform_Location and Onboard_Wifi_Service:")
    print(df_imputed[['Platform_Location', 'Onboard_Wifi_Service']].isna().sum())

    return df_imputed


def impute_using_clustering(df, n_clusters=5, cluster_features=None, features_to_impute=None):
    """
    Impute missing values using KMeans clustering and mode imputation based on clusters.

    Parameters:
    - df (pd.DataFrame): DataFrame with missing values in specified columns.
    - n_clusters (int): Number of clusters for KMeans.
    - cluster_features (list): Features to use for clustering.
    - features_to_impute (list): Features to impute within each cluster.

    Returns:
    - pd.DataFrame: DataFrame with imputed values in specified columns.
    """
    # Define default features for clustering and imputation if not provided
    if cluster_features is None:
        cluster_features = ['Seat_Class', 'Travel_Class', 'Overall_Experience', 'Platform_Location', 'Onboard_Wifi_Service']
    if features_to_impute is None:
        features_to_impute = ['Onboard_Entertainment', 'Legroom', 'Baggage_Handling', 'CheckIn_Service']
    
    # Step 1: Check if all cluster features are in the dataframe
    available_cluster_features = [feature for feature in cluster_features if feature in df.columns]
    missing_features = set(cluster_features) - set(available_cluster_features)
    if missing_features:
        print(f"Warning: The following cluster features are missing and will be excluded from clustering: {missing_features}")

    # Prepare data and encode categorical features for clustering
    df_imputed = df.copy()
    df_cluster_train = df_imputed[available_cluster_features].dropna()
    label_encoders = {}
    
    for col in available_cluster_features:
        if df_cluster_train[col].dtype == 'object':
            le = LabelEncoder()
            df_cluster_train[col] = le.fit_transform(df_cluster_train[col])
            label_encoders[col] = le  # Store encoders for decoding later if needed

    # Step 2: Apply KMeans clustering if we have enough features
    if available_cluster_features:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(df_cluster_train)
        df_imputed.loc[df_cluster_train.index, 'Cluster'] = clusters
    
        # Step 3: Impute missing values for each feature based on cluster modes
        for feature in features_to_impute:
            for cluster in range(n_clusters):
                cluster_indices = df_imputed[df_imputed['Cluster'] == cluster].index
                mode_value = df_imputed.loc[cluster_indices, feature].dropna().mode()
                if not mode_value.empty:
                    df_imputed.loc[cluster_indices, feature] = df_imputed.loc[cluster_indices, feature].fillna(mode_value.iloc[0])

        # Drop the temporary 'Cluster' column
        df_imputed.drop(columns=['Cluster'], inplace=True)
    else:
        print("No cluster features are available for clustering. Skipping KMeans imputation.")

    # Step 4: Fallback for remaining missing values with global mode imputation
    for feature in features_to_impute:
        df_imputed[feature].fillna(df_imputed[feature].mode()[0], inplace=True)
    
    # Verification of missing values after imputation
    print("Missing values after clustering-based imputation and fallback:")
    print(df_imputed[features_to_impute].isna().sum())
    
    # Return the imputed dataframe
    return df_imputed


def impute_online_features(df, features_to_impute=None, cluster_features=None, n_clusters=5):
    """
    Impute missing values using KMeans clustering based on representative features.

    Parameters:
    - df (pd.DataFrame): DataFrame with missing values in specified columns.
    - features_to_impute (list): Features to impute within each cluster.
    - cluster_features (list): Features to use for clustering.
    - n_clusters (int): Number of clusters for KMeans.

    Returns:
    - pd.DataFrame: DataFrame with imputed values in specified columns.
    """
    # Define default features for clustering and imputation if not provided
    if cluster_features is None:
        cluster_features = ['Seat_Class', 'Travel_Class', 'Overall_Experience', 'Platform_Location', 'Onboard_Wifi_Service']
    if features_to_impute is None:
        features_to_impute = ['Online_Support', 'Ease_of_Online_Booking', 'Log_Age']
    
    # Step 1: Check if all cluster features are in the dataframe
    available_cluster_features = [feature for feature in cluster_features if feature in df.columns]
    missing_features = set(cluster_features) - set(available_cluster_features)
    if missing_features:
        print(f"Warning: The following cluster features are missing and will be excluded from clustering: {missing_features}")

    # Prepare data and encode categorical features for clustering
    df_imputed = df.copy()
    df_cluster_train = df_imputed[available_cluster_features].dropna()
    label_encoders = {}
    
    for col in available_cluster_features:
        if df_cluster_train[col].dtype == 'object':
            le = LabelEncoder()
            df_cluster_train[col] = le.fit_transform(df_cluster_train[col])
            label_encoders[col] = le  # Store encoders for decoding later if needed

    # Step 2: Apply KMeans clustering if we have enough features
    if available_cluster_features:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(df_cluster_train)
        df_imputed.loc[df_cluster_train.index, 'Cluster'] = clusters
    
        # Step 3: Impute missing values for each feature based on cluster modes
        for feature in features_to_impute:
            for cluster in range(n_clusters):
                cluster_indices = df_imputed[df_imputed['Cluster'] == cluster].index
                mode_value = df_imputed.loc[cluster_indices, feature].dropna().mode()
                if not mode_value.empty:
                    df_imputed.loc[cluster_indices, feature] = df_imputed.loc[cluster_indices, feature].fillna(mode_value.iloc[0])

        # Drop the temporary 'Cluster' column
        df_imputed.drop(columns=['Cluster'], inplace=True)
    else:
        print("No cluster features are available for clustering. Skipping KMeans imputation.")

    # Step 4: Fallback for remaining missing values with global mode imputation
    for feature in features_to_impute:
        df_imputed[feature].fillna(df_imputed[feature].mode()[0], inplace=True)
    
    # Verification of missing values after imputation
    print("Missing values after clustering-based imputation and fallback:")
    print(df_imputed[features_to_impute].isna().sum())
    
    # Return the imputed dataframe
    return df_imputed

# Import necessary libraries
from lightgbm import LGBMClassifier

# Function to handle one-hot encoding and return the columns used
def one_hot_encode(df, columns, all_columns=None):
    if all_columns is None:
        encoded_df = pd.get_dummies(df, columns=columns)
        return encoded_df, encoded_df.columns
    else:
        encoded_df = pd.get_dummies(df, columns=columns)
        for col in all_columns:
            if col not in encoded_df.columns:
                encoded_df[col] = 0
        return encoded_df[all_columns]

# Step 1: Impute 'Customer_Type' using LightGBM
def impute_customer_type(df):
    customer_features = ['Age_Group', 'Travel_Class', 'Travel_Distance', 
                     'Platform_Location', 'Online_Boarding', 'Catering', 
                     'Ease_of_Online_Booking', 'Onboard_Wifi_Service', 
                     'CheckIn_Service']
    
    df_train = df.dropna(subset=['Customer_Type'])

    # One-hot encoding
    X, all_columns = one_hot_encode(df_train[customer_features], customer_features)
    y = df_train['Customer_Type'].factorize()[0]

    # Model definition
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, device="gpu")

    # Training
    model.fit(X, y)

    # Predict missing 'Customer_Type'
    df_missing = df[df['Customer_Type'].isna()]
    if not df_missing.empty:
        X_missing = one_hot_encode(df_missing[customer_features], customer_features, all_columns=all_columns)
        predictions = model.predict(X_missing)
        # Map predictions back to original categories if needed
        categories = pd.Series(df_train['Customer_Type'].factorize()[1])
        df.loc[df_missing.index, 'Customer_Type'] = categories[predictions].values

# Step 2: Impute 'Type_Travel' using LightGBM
def impute_type_travel(df):
    travel_features = ['Age', 'Customer_Type', 'Travel_Class', 'Travel_Distance',
                        'Platform_Location', 'Ease_of_Online_Booking',
                        'Online_Boarding', 'Catering', 'Onboard_Entertainment',
                        'Onboard_Wifi_Service', 'CheckIn_Service', 'Legroom', 'Seat_Comfort']

    df_train = df.dropna(subset=['Type_Travel'])

    # One-hot encoding
    X, all_columns = one_hot_encode(df_train[travel_features], travel_features)
    y = df_train['Type_Travel'].factorize()[0]

    # Model definition
    model = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, device="gpu")

    # Training
    model.fit(X, y)

    # Predict missing 'Type_Travel'
    df_missing = df[df['Type_Travel'].isna()]
    if not df_missing.empty:
        X_missing = one_hot_encode(df_missing[travel_features], travel_features, all_columns=all_columns)
        predictions = model.predict(X_missing)
        # Map predictions back to original categories if needed
        categories = pd.Series(df_train['Type_Travel'].factorize()[1])
        df.loc[df_missing.index, 'Type_Travel'] = categories[predictions].values

# Step 3: Fill missing values with mode
def fill_missing_with_mode(df, column):
    mode_value = df[column].mode()[0]
    df[column].fillna(mode_value, inplace=True)
    return df
###################### Not Used ###########################
def impute_simpler_features(df):
    print("Imputing 'Catering' with mode...")
    df = fill_missing_with_mode(df, 'Catering')

    print("Imputing 'Onboard_Service' with mode...")
    df = fill_missing_with_mode(df, 'Onboard_Service')
    return df
###########################################################
# Helper function for mode imputation within clusters
def impute_mode_within_clusters(df, feature, cluster_col):
    for cluster in df[cluster_col].dropna().unique():
        cluster_indices = df[df[cluster_col] == cluster].index
        cluster_data = df.loc[cluster_indices, feature].dropna()
        
        if not cluster_data.empty:
            mode_value = cluster_data.mode()
            if not mode_value.empty:
                df.loc[cluster_indices, feature] = df.loc[cluster_indices, feature].fillna(mode_value[0])
    return df


# Step 10: Impute 'Arrival_Time_Convenient'
def impute_arrival_time_convenient(df):
    arrival_features = ['Platform_Location', 'Ease_of_Online_Booking', 'Travel_Class', 'Customer_Type']
    df_arrival_train = df[arrival_features].dropna()

    # One-hot encode categorical features
    df_arrival_train_encoded = pd.get_dummies(df_arrival_train)

    # Apply KMeans clustering
    kmeans_arrival = KMeans(n_clusters=4, random_state=42)
    df.loc[df_arrival_train.index, 'Arrival_Cluster'] = kmeans_arrival.fit_predict(df_arrival_train_encoded)

    # Impute using mode within clusters
    df = impute_mode_within_clusters(df, 'Arrival_Time_Convenient', 'Arrival_Cluster')
    df.drop(columns=['Arrival_Cluster'], inplace=True)
    print("Imputed 'Arrival_Time_Convenient' using mode within clusters.")
    return df

# Step 11: Impute 'Catering'
def impute_catering(df):
    catering_features = ['Onboard_Service', 'Seat_Comfort', 
                     'Travel_Class', 'Onboard_Entertainment', 'Customer_Type', 
                     'CheckIn_Service']

    df_catering_train = df[catering_features].dropna()

    # One-hot encode categorical features
    df_catering_train_encoded = pd.get_dummies(df_catering_train)

    # Apply KMeans clustering
    kmeans_catering = KMeans(n_clusters=4, random_state=42)
    df.loc[df_catering_train.index, 'Catering_Cluster'] = kmeans_catering.fit_predict(df_catering_train_encoded)

    # Impute using mode within clusters
    df = impute_mode_within_clusters(df, 'Catering', 'Catering_Cluster')
    df.drop(columns=['Catering_Cluster'], inplace=True)
    print("Imputed 'Catering' using mode within clusters.")
    return df

# Step 12: Impute 'Onboard_Service'
def impute_onboard_service(df):
    onboard_features = ['Onboard_Entertainment', 'Onboard_Wifi_Service', 'CheckIn_Service']
    df_onboard_train = df[onboard_features].dropna()

    # One-hot encode categorical features
    df_onboard_train_encoded = pd.get_dummies(df_onboard_train)

    # Apply KMeans clustering
    kmeans_onboard = KMeans(n_clusters=4, random_state=42)
    df.loc[df_onboard_train.index, 'Onboard_Cluster'] = kmeans_onboard.fit_predict(df_onboard_train_encoded)

    # Impute using mode within clusters
    df = impute_mode_within_clusters(df, 'Onboard_Service', 'Onboard_Cluster')
    df.drop(columns=['Onboard_Cluster'], inplace=True)
    print("Imputed 'Onboard_Service' using mode within clusters.")
    return df


def data_pre_processing():
    print("Starting Data Preprocessing...")

    # Load saved datasets from the EDA
    train_data = pd.read_csv("Data/train_combined_data.csv")
    test_data = pd.read_csv("Data/test_combined_data.csv")
    datasets = {
        "Training": train_data,
        "Test": test_data
    }
    for name, data in datasets.items():
        print(f"\nMissing Values - {name}:")
        print(data.isnull().sum())
        print(f"Duplicate Rows in {name}:", data.duplicated().sum())
    print("Columns in train:", train_data.columns)
    print("Columns in test:", test_data.columns)
    
    # Drop 'Log_Age' from both datasets
    if 'Log_Age' in train_data.columns:
        train_data.drop(columns=['Log_Age'], inplace=True)
        print("Dropped 'Log_Age' from train dataset.")
    if 'Log_Age' in test_data.columns:
        test_data.drop(columns=['Log_Age'], inplace=True)
        print("Dropped 'Log_Age' from test dataset.")

    # Step 1: Process Age column
    print("\nProcessing Age column...")
    train_data = impute_and_bin_age(train_data, strategy='median')
    test_data = impute_and_bin_age(test_data, strategy='median')

    # Step 2: Process Gender column based on Age_Group
    print("\nProcessing Gender column...")
    train_data = impute_gender_based_on_age_group(train_data)
    test_data = impute_gender_based_on_age_group(test_data)

    # Step 3: Impute Log Delays with Regression
    print("\nProcessing Delay columns with regression imputation...")
    train_data = impute_log_delays_with_regression(train_data)
    test_data = impute_log_delays_with_regression(test_data)

    # Step 4: Impute Ordinal Ratings by Group (example: 'Seat_Comfort' grouped by 'Travel_Class')
    print("\nProcessing Ordinal Ratings by Group...")
    train_data = impute_ordinal_by_group(train_data, column='Seat_Comfort', group_by_col='Travel_Class')
    test_data = impute_ordinal_by_group(test_data, column='Seat_Comfort', group_by_col='Travel_Class')

    # # Step 5: Impute 'Seat_Comfort' using KNN
    # print("\nImputing Seat_Comfort with KNN...")
    # train_data = knn_impute_seat_comfort(train_data, knn_features=['Seat_Class', 'Gender', 'Age_Group', 'Cleanliness', 'Travel_Class', 'Seat_Comfort'])
    # test_data = knn_impute_seat_comfort(test_data, knn_features=['Seat_Class', 'Gender', 'Age_Group', 'Cleanliness', 'Travel_Class', 'Seat_Comfort'])

    # Step 6: Impute 'Platform_Location' and 'Onboard_Wifi_Service' using mode and KNN
    print("\nImputing Platform_Location and Onboard_Wifi_Service with combined mode and KNN...")
    train_data = impute_platform_location_and_wifi(train_data, knn_features=['Platform_Location', 'Onboard_Wifi_Service', 'Travel_Class', 'Seat_Class', 'Onboard_Entertainment'])
    test_data = impute_platform_location_and_wifi(test_data, knn_features=['Platform_Location', 'Onboard_Wifi_Service', 'Travel_Class', 'Seat_Class', 'Onboard_Entertainment'])

    # Step 7: Clustering-based imputation for specific features
    print("\nImputing with clustering for multiple columns...")
    features_to_impute = ['Onboard_Entertainment', 'Legroom', 'Baggage_Handling', 'CheckIn_Service']
    cluster_features = ['Seat_Class', 'Travel_Class', 'Overall_Experience', 'Platform_Location', 'Onboard_Wifi_Service']
    train_data = impute_using_clustering(train_data, n_clusters=5, cluster_features=cluster_features, features_to_impute=features_to_impute)
    test_data = impute_using_clustering(test_data, n_clusters=5, cluster_features=cluster_features, features_to_impute=features_to_impute)

   # Step 8: Online feature imputation
    print("\nImputing online-related features...")
    features_to_impute = ['Online_Support', 'Ease_of_Online_Booking', 'Online_Boarding',]
    cluster_features = ['Seat_Class', 'Travel_Class', 'Overall_Experience', 'Platform_Location', 'Onboard_Wifi_Service']
    train_data = impute_online_features(train_data, features_to_impute=features_to_impute, cluster_features=cluster_features)
    test_data = impute_online_features(test_data, features_to_impute=features_to_impute, cluster_features=cluster_features)

    print("Columns in train:", train_data.columns)
    print("Columns in test:", test_data.columns)

    # Step 9 : (e.g., impute_customer_type, impute_type_travel, etc.)
    print("\nProcessing Customer_Type...")
    impute_customer_type(train_data)
    impute_customer_type(test_data)
    print("\nProcessing Type_Travel...")
    impute_type_travel(train_data)
    impute_type_travel(test_data)
    print("\nProcessing cleanliness...")
    train_data = fill_missing_with_mode(train_data,"Cleanliness")
    test_data = fill_missing_with_mode(test_data,"Cleanliness")

    # Step 10,11,12
    # Apply the functions
    print("\nProcessing arrival_time_convenient ...")
    train_data = impute_arrival_time_convenient(train_data)
    print("\nProcessing catering ...")
    train_data = impute_catering(train_data)
    print("\nProcessing onboard service ...")
    train_data = impute_onboard_service(train_data)
    print("\nProcessing Test df remaining features...")
    test_data = impute_arrival_time_convenient(test_data)
    test_data = impute_catering(test_data)
    test_data = impute_onboard_service(test_data)
    #Catering remains so : 
    train_data = fill_missing_with_mode(train_data,"Catering")
    test_data = fill_missing_with_mode(test_data,"Catering")

    # Check for remaining missing data and print summary
    print("\n--- Summary of Missing Data After Processing ---")
    missing_data_train = train_data.isnull().sum()[train_data.isnull().sum() > 0]
    missing_data_test = test_data.isnull().sum()[test_data.isnull().sum() > 0]

    if missing_data_train.empty:
        print("No missing data remaining in the train dataset.")
    else:
        print("Remaining missing data in train dataset:")
        print(missing_data_train)

    if missing_data_test.empty:
        print("No missing data remaining in the test dataset.")
    else:
        print("Remaining missing data in test dataset:")
        print(missing_data_test)

    # Save the preprocessed data for further analysis or modeling
    print("this col: onboard_entertainment acts unusual when i save the csv , i found out that 18 rows filled with 'nan',and i have not any missing data!!")
    print(train_data['Onboard_Entertainment'].describe())
    unique_values = train_data['Onboard_Entertainment'].unique()
    print(unique_values)
    value_counts = train_data['Onboard_Entertainment'].value_counts()
    print(value_counts)
    # Assuming train_data is your DataFrame
    # Identify the rows with 'nan' in the 'Onboard_Entertainment' column
    nan_rows_train = train_data[train_data['Onboard_Entertainment'] == 'nan']

    # Print the identified rows for verification
    print("Rows with 'nan' values in 'Onboard_Entertainment':")
    print(nan_rows_train)

    # Group by 'Onboard_Service' and calculate the mode for 'Onboard_Entertainment'
    mode_by_group_train = train_data.groupby('Onboard_Service')['Onboard_Entertainment'].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    mode_by_group_test = test_data.groupby('Onboard_Service')['Onboard_Entertainment'].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)

    # Replace 'nan' values with the mode within each group
    def replace_nan_with_mode_train(row):
        if row['Onboard_Entertainment'] == 'nan':
            return mode_by_group_train[row['Onboard_Service']]
        else:
            return row['Onboard_Entertainment']
        
        # Replace 'nan' values with the mode within each group
    def replace_nan_with_mode_test(row):
        if row['Onboard_Entertainment'] == 'nan':
            return mode_by_group_test[row['Onboard_Service']]
        else:
            return row['Onboard_Entertainment']

    train_data['Onboard_Entertainment'] = train_data.apply(replace_nan_with_mode_train, axis=1)
    test_data['Onboard_Entertainment'] = test_data.apply(replace_nan_with_mode_test, axis=1)

    # Verify the unique values and their counts in 'Onboard_Entertainment' column
    print(train_data['Onboard_Entertainment'].unique())
    print(train_data['Onboard_Entertainment'].value_counts())



    train_data.to_csv("Data/train_preprocessed.csv", index=False,encoding='utf-8')
    test_data.to_csv("Data/test_preprocessed.csv", index=False,encoding='utf-8')
    
    print("\nData Preprocessing Completed. Preprocessed files saved as train_preprocessed.csv and test_preprocessed.csv")
    print(train_data.info())
    print(test_data.info())
