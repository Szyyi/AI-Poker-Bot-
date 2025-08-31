import os
import numpy as np
import pandas as pd
from scipy.stats import entropy
import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy, set_global_policy
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# GPU Configuration
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
set_global_policy('mixed_float16')

# Define the data path
# Define the root and paths
root = "./phase1"
data_path = f"{root}/data"
model_root = f"{root}/submission/BT3"

# Ensure the raw and processed directories exist
raw_path = f"{data_path}/raw"
processed_path = f"{data_path}/processed"
os.makedirs(processed_path, exist_ok=True)

# Reading the poker hand training dataset from the specified file path
df_hands_train = pd.read_csv(
    f"{raw_path}/poker-hand-training-true.data",
    names=["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "Hand Rank"]
)

# Displaying the first few rows of the training dataset to verify successful import
print("Training Data Preview:")
print(df_hands_train.head())

# Reading the 10-million hands dataset for advanced analysis or optional use
df_hands_10m = pd.read_csv(
    f"{raw_path}/poker10m",
    names=["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "Hand Rank"]
)

# Displaying the first few rows of the 10-million dataset
print("10-Million Hands Data Preview:")
print(df_hands_10m.head())

# Reading the poker hand testing dataset for evaluation
df_hands_test = pd.read_csv(
    f"{raw_path}/poker-hand-testing.data",
    names=["S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5", "Hand Rank"]
)

# Displaying the first few rows of the testing dataset to verify successful import
print("Testing Data Preview:")
print(df_hands_test.head())


# Selecting all columns except the last one (Hand Rank) to create the feature set (X_train)
# reset_index(drop=True) ensures the index is reset and the original index is not retained
X_train = df_hands_train.iloc[:, :-1].reset_index(drop=True)

# Selecting the last column (Hand Rank) as the label set (y_train)
y_train = df_hands_train.iloc[:, -1]

# Displaying the first 5 rows of the feature set to verify the selection
X_train.head(5)


X_test = df_hands_test.iloc[:, :-1].reset_index(drop=True)
y_test = df_hands_test.iloc[:, -1]
X_test.head(5)



def get_model_log(model, xtest, ytest, model_name= 'model'):
    """
    Trains a machine learning model and evaluates its performance metrics.

    Args:
       model: The machine learning model to train and evaluate.
       xtest (pd.DataFrame): The testing data features.
       ytest (pd.Series): The testing data target labels.
       model_name (str, optional): The name of the model. Defaults to 'model'.

    Returns:
        pd.DataFrame: A DataFrame containing the model's performance metrics:
            - Accuracy
            - F1 score (weighted average)
            - Precision (weighted average)
            - Recall (weighted average)
    """

    # Generate predictions using the trained model on the test data
    y_pred = model.predict(xtest)

    # Handle LSTM model predictions (if the model name includes 'lstm')
    if 'lstm' in model_name.lower():
        # Print the probability distribution for LSTM predictions
        print(f"LSTM pred probability: {y_pred}")
        # Convert probability predictions to class labels by selecting the highest probability
        y_pred = np.argmax(y_pred, axis=1)
        print(f"LSTM prediction hand: {y_pred}")

    # Create a DataFrame to store the performance metrics
    model_log = pd.DataFrame(columns=["Accuracy", "F1", "Precision", "Recall"])

    # Calculate and log the accuracy score
    model_log.loc[0, "Accuracy"] = accuracy_score(ytest, y_pred)
    # Calculate and log the F1 score (weighted average)
    model_log.loc[0, "F1"] = f1_score(ytest, y_pred, average="weighted")
    # Calculate and log the precision score (weighted average)
    model_log.loc[0, "Precision"] = precision_score(ytest, y_pred, average="weighted")
    # Calculate and log the recall score (weighted average)
    model_log.loc[0, "Recall"] = recall_score(ytest, y_pred, average="weighted")

    # Return the DataFrame containing the performance metrics
    return model_log


# Load training data
X_train = pd.read_csv(f"{data_path}/raw/poker-hand-training-true.data", header=None)
X_train.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']

def sort_hand(df):
    suits = df[['S1', 'S2', 'S3', 'S4', 'S5']].values
    ranks = df[['C1', 'C2', 'C3', 'C4', 'C5']].values
    sorted_indices = np.argsort(ranks, axis=1)
    sorted_suits = np.take_along_axis(suits, sorted_indices, axis=1)
    sorted_ranks = np.take_along_axis(ranks, sorted_indices, axis=1)
    df[['S1', 'S2', 'S3', 'S4', 'S5']] = sorted_suits
    df[['C1', 'C2', 'C3', 'C4', 'C5']] = sorted_ranks
    return df

def get_rank_patterns(ranks):
    sorted_ranks = np.sort(ranks, axis=1)
    rank_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=14)[1:], axis=1, arr=ranks)
    gaps = np.diff(sorted_ranks, axis=1)
    is_straight = np.zeros(len(ranks), dtype=bool)
    for i in range(len(ranks)):
        hand = sorted_ranks[i]
        if (np.all(np.diff(hand) == 1) or
                (hand[0] == 1 and np.array_equal(hand[1:], [2, 3, 4, 5])) or
                (hand[0] == 1 and np.array_equal(hand[1:], [10, 11, 12, 13]))):
            is_straight[i] = True
    rank_values = np.arange(1, 14)
    keys = rank_counts * 100 + rank_values
    sorted_indices = np.argsort(-keys, axis=1)
    sorted_ranks_with_counts = sorted_indices + 1
    return {
        'rank_counts': rank_counts,
        'pairs': np.sum(rank_counts == 2, axis=1),
        'trips': np.sum(rank_counts == 3, axis=1),
        'quads': np.sum(rank_counts == 4, axis=1),
        'is_straight': is_straight,
        'gaps': gaps,
        'sorted_ranks': sorted_ranks,
        'primary_rank': sorted_ranks_with_counts[:, 0],
        'primary_count': rank_counts[np.arange(len(rank_counts)), sorted_indices[:, 0]],
        'secondary_rank': sorted_ranks_with_counts[:, 1],
        'secondary_count': rank_counts[np.arange(len(rank_counts)), sorted_indices[:, 1]]
    }

def get_suit_patterns(suits):
    suit_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=5)[1:], axis=1, arr=suits)
    return {
        'suit_counts': suit_counts,
        'is_flush': np.max(suit_counts, axis=1) == 5,
        'suit_distribution': np.apply_along_axis(lambda x: len(np.unique(x)), axis=1, arr=suits)
    }

def detect_hand_strength(rank_patterns, suit_patterns):
    sorted_ranks = rank_patterns['sorted_ranks']
    is_straight = rank_patterns['is_straight']
    is_flush = suit_patterns['is_flush']
    is_straight_flush = np.logical_and(is_straight, is_flush)
    is_royal = np.logical_and.reduce([
        is_straight_flush,
        sorted_ranks[:, 0] == 1,
        sorted_ranks[:, -1] == 13
    ])
    return pd.DataFrame({
        'has_straight': is_straight.astype(int),
        'has_flush': is_flush.astype(int),
        'has_straight_flush': is_straight_flush.astype(int),
        'has_royal_flush': is_royal.astype(int)
    })

def extract_additional_features(df, rank_patterns, suit_patterns):
    ranks = df[['C1', 'C2', 'C3', 'C4', 'C5']].values
    suit_counts = suit_patterns['suit_counts']
    return pd.DataFrame({
        'rank_entropy': np.apply_along_axis(
            lambda x: entropy(x[x > 0], base=2) if np.sum(x) > 0 else 0,
            1, rank_patterns['rank_counts']
        ),
        'unique_suits': suit_patterns['suit_distribution']
    })

def pre_process_data(data):
    df = data.copy()
    df = sort_hand(df)
    ranks = df[['C1', 'C2', 'C3', 'C4', 'C5']].values
    suits = df[['S1', 'S2', 'S3', 'S4', 'S5']].values
    rank_patterns = get_rank_patterns(ranks)
    suit_patterns = get_suit_patterns(suits)
    strength_features = detect_hand_strength(rank_patterns, suit_patterns)
    advanced_features = extract_additional_features(df, rank_patterns, suit_patterns)
    df = pd.concat([df, strength_features.add_prefix('strength_'), advanced_features.add_prefix('feat_')], axis=1)
    return df

# Preprocess and save X_train
X_train_pre = pre_process_data(X_train)
X_train_pre.to_csv(f"{data_path}/processed/X_train_pre.csv", index=False)

print("Preprocessing complete. Processed data saved to:", f"{data_path}/processed/X_train_pre.csv")
