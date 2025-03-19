import os
import pandas as pd
import numpy as np
import math
from pandas import DataFrame
from typing import Tuple, List

from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


def csv_to_parquet(folder_path, path_to_save, overwrite=False):
    # Get list of CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if len(csv_files) == 0:
        print("No CSV files found.")
        return

    # Convert each CSV file to Parquet if the corresponding Parquet file doesn't exist
    for csv_file in csv_files:
        csv_path = os.path.join(folder_path, csv_file)
        parquet_path = os.path.join(path_to_save, csv_file.replace('.csv', '.parquet'))

        # Check if the Parquet file already exists
        if not os.path.exists(parquet_path) or overwrite:
            try:
                # Read the CSV file
                df = pd.read_csv(csv_path)
            except UnicodeDecodeError:
                # Handle encoding issues
                df = pd.read_csv(csv_path, encoding='ISO-8859-1')

            # Save as Parquet with the same name as the CSV but .parquet extension
            df.to_parquet(parquet_path)
            print(f"Converted {csv_file} to {parquet_path}")
        else:
            print(f"Parquet file for {csv_file} already exists and will not be overwritten.")


def count_duplicated_rows(
        df: pd.DataFrame) -> None:
    """
    Count and print the number of duplicated rows in a DataFrame
    (based on all columns).
    """
    num_duplicated_rows = df.duplicated().sum()
    print(f"The DataFrame contains {num_duplicated_rows} duplicated rows.")


def missing_percentage(
        df: pd.DataFrame) -> DataFrame:
    """
    Calculate the percentage of missing values in a DataFrame.
    :param df: pandas DataFrame    :return:
    """
    total_missing = df.isnull().sum()
    percent_of_missing = total_missing / df.isnull().count() * 100
    concat_missing = pd.concat(
        [total_missing, percent_of_missing],
        axis=1,
        keys=['Total_missing', 'Percent_missing']
    ).sort_values(by=["Percent_missing"], ascending=False)
    return concat_missing


def reduce_memory_usage_pd(
        df: pd.DataFrame,
        verbose: bool = True) -> pd.DataFrame:
    """Optimize memory usage of a pandas DataFrame."""

    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_numeric_dtype(col_type):
            if pd.api.types.is_integer_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif pd.api.types.is_float_dtype(col_type):
                max_val = df[col].max()
                if max_val < np.finfo(np.float32).max:  # Check if max value fits in float32
                    df[col] = df[col].astype(np.float32)  # Downcast to float32

        elif pd.api.types.is_object_dtype(col_type):
            num_unique = df[col].nunique()
            num_total = len(df[col])

            # Convert strings to categorical if unique values are much less than total
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose:
        print(f"Memory before: {start_mem:.2f} MB. \n"
              f"Memory after: {end_mem:.2f} MB.\n"
              f"Percent of reduction: ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")

    return df


def reduce_features(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        target_col: str,
        features_to_remove: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if target_col not in df_train.columns:
        raise ValueError(f"Features {target_col} not found in Train DataFrame.")

    # Drop the features without checking if they exist (ignores missing columns)
    df_train_reduced = df_train.drop(columns=features_to_remove, errors='ignore')
    df_test_reduced = df_test.drop(columns=features_to_remove, errors='ignore')

    return df_train_reduced, df_test_reduced


def transform_columns(
        df: pd.DataFrame,
        target_columns: list,
        return_columns: list,
        remove_old_cols: bool = False) -> pd.DataFrame:
    """
    Transforms the specified target columns in the DataFrame by converting them to years
    and adding them as new columns with the specified return column names. Optionally removes old columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to transform.
    target_cols (list): List of columns to be transformed (in days).
    return_cols (list): List of new column names to store the transformed data (in years).
    remove_old_cols (bool): If True, removes the original target columns after transformation.

    Returns:
    pd.DataFrame: The DataFrame with the new transformed columns.
    """
    # Ensure the length of target_cols and return_cols match
    if len(target_columns) != len(return_columns):
        raise ValueError("The number of target columns and return columns must be the same.")

    # Iterate through target columns and return column names
    for target_col, return_col in zip(target_columns, return_columns):
        df[return_col] = round(abs(df[target_col] / 365))

    # Optionally remove the old columns
    if remove_old_cols:
        df = df.drop(columns=target_columns, errors='ignore')

    return df


def quantile_cap(df: pd.DataFrame, target_col: str, quantile: float) -> pd.DataFrame:
    """
    Caps the values in the target column at the specified quantile.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_col (str): The column in which values will be capped.
    quantile (float): The quantile threshold (between 0 and 1) to cap values.

    Returns:
    pd.DataFrame: DataFrame with the capped values in the target column.
    """

    # Calculate the quantile cap value
    cap = df[target_col].quantile(quantile)

    # Cap values in the target column
    df[target_col] = np.where(df[target_col] > cap, cap, df[target_col])

    return df


# Cramer's V calculation
def cramers_v(chi2, n, dof):
    """
    Cramer's V is calculated for categorical features when running the Chi-Square test
    to assess the strength of association.
    """
    return np.sqrt(chi2 / (n * dof))


# Cohen's d calculation for binary target
def cohens_d(group1, group2):
    """
    Cohen's d is calculated for numerical features in binary target cases using a t-test,
    measuring the effect size between two groups (target = 0 and target = 1).
    """
    diff_means = np.mean(group1) - np.mean(group2)
    pooled_var = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)
    return diff_means / pooled_var


def hypothesis_testing(df, target, categorical_features, numerical_features, alpha = 0.05):
    """
    Perform hypothesis testing on categorical and numerical features to determine their
    significance with respect to the target variable. The function conducts Chi-Square
    tests for categorical features, and t-tests (or ANOVA) for numerical features, returning
    two dataframes with the test results, p-values, and effect sizes (Cramer's V and Cohen's d).

    Parameters:
    ----------
    df (pd.DataFrame): The input dataset containing the features and the target variable.

    target (str) : The name of the target variable column in the dataset.

    categorical_features (list): List of column names representing the categorical features to be tested.

    numerical_features (list): List of column names representing the numerical features to be tested.

    alpha (float), optional (default=0.05): The significance level for the hypothesis tests.
    If the p-value is less than alpha, the null hypothesis is rejected.

    Returns:
    -------
    categorical_df (pd.DataFrame): A dataframe containing the results of hypothesis tests on categorical features,
    including p-values, test type (Chi-Square), and Cramer's V for association strength.

    numerical_df (pd.DataFrame): A dataframe containing the results of hypothesis tests on numerical features,
    including p-values, test type (T-test/ANOVA), and Cohen's d for effect size.
    """
    # Lists to store results for categorical and numerical features
    categorical_results = []
    numerical_results = []

    # 1. Categorical Features - Chi-Square Test with Cramer"s V
    for feature in categorical_features:
        # Create a contingency table
        contingency_table = pd.crosstab(df[feature], df[target])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        reject_null = "Yes" if p < alpha else "No"
        n = contingency_table.sum().sum()  # Total sample size
        cramer_v = cramers_v(chi2, n, dof)
        categorical_results.append(
            {"Feature": feature, "P-value": p, "Test": "Chi-Square", "Reject Null Hypothesis": reject_null,
             "Cramer\"s V": cramer_v})

    # 2. Numerical Features - T-test/ANOVA with Cohen"s d
    if df[target].nunique() == 2:
        # Binary Target - T-test and Cohen"s d
        for feature in numerical_features:
            group1 = df[df[target] == 0][feature]
            group2 = df[df[target] == 1][feature]
            t_stat, p = ttest_ind(group1, group2, nan_policy="omit")
            reject_null = "Yes" if p < alpha else "No"
            cohen_d = cohens_d(group1, group2)
            numerical_results.append(
                {"Feature": feature, "P-value": p, "Test": "T-test", "Reject Null Hypothesis": reject_null,
                 "Cohen\"s d": cohen_d})
    else:
        # Multi-class Target - ANOVA (Cohen"s d is not appropriate for multi-class)
        for feature in numerical_features:
            groups = [df[df[target] == category][feature] for category in df[target].unique()]
            f_stat, p = f_oneway(*groups)
            reject_null = "Yes" if p < alpha else "No"
            numerical_results.append(
                {"Feature": feature, "P-value": p, "Test": "ANOVA", "Reject Null Hypothesis": reject_null,
                 "Cohen\"s d": np.nan})  # Cohen"s d only valid for binary targets

    # Convert results to DataFrames
    categorical_df = pd.DataFrame(categorical_results)
    numerical_df = pd.DataFrame(numerical_results)

    # Sort both DataFrames by "Reject Null Hypothesis" and "P-value"
    categorical_df = categorical_df.sort_values(by=["Reject Null Hypothesis", "P-value"], ascending=[False, True])
    numerical_df = numerical_df.sort_values(by=["Reject Null Hypothesis", "P-value"], ascending=[False, True])

    return categorical_df, numerical_df


def age_year_bins(df: pd.DataFrame, target_feature: str) -> pd.DataFrame:
    """
    Bin age values into predefined groups.

    Args:
        df: The input DataFrame.
        target_feature: The name of the column containing age values in years.

    Returns:
        Series with binned age values.
    """
    bins = [0, 18, 25, 35, 45, 55, 65, np.inf]
    labels = ["0-18", "19-25", "26-35", "36-45", "46-55", "55-65", "65+"]

    return pd.cut(df[target_feature], bins=bins, labels=labels)


def numeric_quantile_bins(df: pd.DataFrame, target_feature: str, bin_number: int = 5) -> pd.Series:
    if df[target_feature].nunique() == 1:
        return pd.Series(["Q1"] * len(df), index=df.index)

    # Generate labels based on the number of bins
    labels = [f"Q{i + 1}" for i in range(min(bin_number, df[target_feature].nunique()))]

    # Apply quantile binning
    return pd.qcut(df[target_feature], q=bin_number, labels=labels, duplicates="drop")


def convert_binary_values_custom(df, binary_columns, custom_mappings = None):
    """
    Converts binary values in specified columns to 0 and 1 based on custom mappings.
    If no custom mappings are provided, handles (Y, N) and (Yes, No) for generic binary columns,
    and (M, F) and (Male, Female) for gender-related columns.

    Args:
    df (pd.DataFrame): The input DataFrame containing binary columns.
    binary_columns (list): A list of column names to convert.
    custom_mappings (dict, optional): A dictionary mapping original values to target values.

    Returns:
    pd.DataFrame: The DataFrame with converted binary values.
    """
    # Define default conversion mappings if custom mappings are not provided
    if custom_mappings is None:
        conversion_map = {
            "Y": 1, "N": 0,
            "Yes": 1, "No": 0,
            "M": 1, "F": 0,
            "Male": 1, "Female": 0
        }
    else:
        conversion_map = custom_mappings

    # Convert specified columns
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].replace(conversion_map)

    return df


def feature_reduction_xgboost(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        threshold: int = 1) -> pd.DataFrame:
    """
    :param x_train: Training data features
    :param y_train: Training data target
    :param threshold: Minimum importance threshold for feature selection
    :return: DataFrame of selected features with their importance scores
    """

    x_train[x_train.select_dtypes(["object"]).columns] = x_train.select_dtypes(["object"]).astype("category")

    # Train model
    model = XGBClassifier(verbosity=1, random_state=42, enable_categorical=True)
    model.fit(x_train, y_train)

    # Get feature importance
    feature_importance = model.get_booster().get_score(importance_type='weight')

    feature_importance_df = pd.DataFrame(
        list(feature_importance.items()),
        columns=["Feature", "Importance"]
    )

    if threshold >= 1:
        feature_importance_df = feature_importance_df[feature_importance_df["Importance"] > threshold]

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=False).reset_index(drop=True)

    return feature_importance_df


def remove_single_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes single values from a DataFrame.
    :param df: Data frame to be cleaned.
    :return: Data frame without single values columns.
    """
    df_cleaned = df.loc[:, df.nunique(dropna=False) > 1]
    return df_cleaned


def train_test_valid_split(x: pd.DataFrame, y: pd.DataFrame, ran_state=42, test_split_size = 0.2, size = 1):
    dataset_size = math.ceil(x.shape[0] * size)
    x = x.iloc[:dataset_size]
    y = y.iloc[:dataset_size]

    x_split, x_test, y_split, y_test = train_test_split(
        x, y,
        test_size=test_split_size,
        random_state=ran_state,
        stratify=y)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_split, y_split,
        test_size=0.25,
        random_state=ran_state,
        stratify=y_split)

    print(f"Training set shape: {x_train.shape}")
    print(f"Validation set shape: {x_valid.shape}")
    print(f"Test set shape: {x_test.shape}")

    return x_train, x_test, x_valid, y_train, y_test, y_valid


def filter_existing_columns(df: pd.DataFrame, column_list: List[str]) -> List[str]:
    """
    Filters the provided list of columns and returns only those that exist in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame in which to check for column existence.
    column_list (List[str]): The list of column names to filter.

    Returns:
    List[str]: A list of column names that are present in the DataFrame.
    """
    return [col for col in column_list if col in df.columns]


def update_and_classify_features(
        df: pd.DataFrame, binary_feature = None, categorical_feature = None, numerical_feature = None):
    """
    Update feature lists based on DataFrame column types and unique value counts.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        binary_feature (list): List to store binary feature names.
        categorical_feature (list): List to store categorical feature names.
        numerical_feature (list): List to store float feature names.

    Returns:
        tuple: Updated lists of feature names (binary_feature, categorical_feature, numerical_feature).
    """
    # Initialize lists if they are None
    if binary_feature is None:
        binary_feature = []
    if categorical_feature is None:
        categorical_feature = []
    if numerical_feature is None:
        numerical_feature = []

    # Classify columns
    for col in df.columns:
        unique_count = df[col].nunique()

        # Classify numerical features
        if (pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(
                df[col])) and col not in numerical_feature:
            numerical_feature.append(col)

        # Classify boolean features
        elif pd.api.types.is_bool_dtype(df[col]) and col not in binary_feature:
            binary_feature.append(col)

        elif unique_count < 50 and (
                pd.api.types.is_string_dtype(df[col]) or
                pd.api.types.is_categorical_dtype(df[col]) or
                pd.api.types.is_object_dtype(df[col])
        ) and col not in categorical_feature:
            df[col] = df[col].astype('category')
            categorical_feature.append(col)

    return df, binary_feature, categorical_feature, numerical_feature
