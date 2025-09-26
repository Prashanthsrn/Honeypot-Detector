import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    test_df = pd.read_csv(test_path, encoding="utf-8-sig")
    
    # Fix BOM in first column
    train_df.rename(columns={train_df.columns[0]: "id"}, inplace=True)
    test_df.rename(columns={test_df.columns[0]: "id"}, inplace=True)
    
    return train_df, test_df

def preprocess_data(train_df, test_df):
    drop_cols = ["id", "attack_cat"]
    
    # Features / target
    X_train = train_df.drop(columns=drop_cols + ["label"])
    y_train = train_df["label"]
    
    X_test = test_df.drop(columns=drop_cols + ["label"])
    y_test = test_df["label"]
    
    # Identify categorical vs numeric
    cat_cols = ["proto", "service", "state"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    
    # Numeric pipeline: median impute + scale
    num_transformer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_train_num = scaler.fit_transform(num_transformer.fit_transform(X_train[num_cols]))
    X_test_num = scaler.transform(num_transformer.transform(X_test[num_cols]))
    
    # Categorical pipeline: fill missing + one-hot encode
    cat_transformer = SimpleImputer(strategy="constant", fill_value="missing")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    X_train_cat = encoder.fit_transform(cat_transformer.fit_transform(X_train[cat_cols]))
    X_test_cat = encoder.transform(cat_transformer.transform(X_test[cat_cols]))
    
    # Combine numeric + categorical
    X_train_prep = np.hstack([X_train_num, X_train_cat])
    X_test_prep = np.hstack([X_test_num, X_test_cat])
    
    return X_train_prep, X_test_prep, y_train.values, y_test.values
