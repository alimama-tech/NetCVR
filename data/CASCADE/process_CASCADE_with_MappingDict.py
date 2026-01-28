import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import hashlib,farmhash
import ast
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
SECONDS_A_DAY = 86400
cate_bin_size = (8,327680,655360,655360,8192,655360,16,32,256,81920,163840,8,512,128,256,1024,128,16,8,8,16,16)
start_date_str = "2025-05-01 00:00:00"
start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
start_timestamp = int(start_date.timestamp())


def calc_seconds_since_start(ts):
    if ts == -1:
        return -1
    try:
        return int(ts) - start_timestamp
    except:
        return -1

def extract_first_from_numeric_array(numeric_val):
    try:
        arr = ast.literal_eval(numeric_val)
        if isinstance(arr, list) and len(arr) > 0:
            return arr[0]
        else:
            return -1
    except (ValueError, SyntaxError):
        return -1
    
def extract_first_from_array(string_val):
    try:
        arr = ast.literal_eval(string_val)

        if isinstance(arr, list) and len(arr) > 0:
            return str(arr[0])
        else:
            return str(string_val)
    except (ValueError, SyntaxError):
        return str(string_val)

def process_features(df):
    df = df.copy()
    print("Processing categorical features...")
    for i in tqdm(range(0, 22), desc="Categorical columns"):
        col = df.columns[i]
        bucket_size = cate_bin_size[i]
        df[col] = df[col].apply(lambda x: stable_hash(x, bucket_size))
    df = df.replace('', np.nan)
    df = df.fillna(0)
    df = df.astype(np.int64)
    return df



def process_features_with_mapping_dict(df):
    df = df.copy()
    print("Processing categorical features...")
    mappings = {}
    for i in tqdm(range(0, 22), desc="Categorical columns"):
        col = df.columns[i]
        bucket_size = cate_bin_size[i]
        unique_vals = df[col].dropna().unique()
        mapping_dict = {}
        for val in unique_vals:
            mapped_id = stable_hash(val, bucket_size)
            mapping_dict[val] = mapped_id
        
        mappings[col] = mapping_dict

        df[col] = df[col].apply(lambda x: stable_hash(x, bucket_size))
    
    df = df.replace('', np.nan)
    df = df.fillna(0)
    df = df.astype(np.int64)
    
    return df, mappings




def stable_hash(val, bucket_size):
    val_str = str(val)
    return int(farmhash.fingerprint64(val_str.encode('utf-8'))) % bucket_size

def preprocess_data_df(data_path,output_path,mapping_output_path):
    print("Loading data from ", data_path)
    data = pd.read_csv(data_path, sep="\t", header=None)

    print("preprocessing data from ", data_path)

    click_ts = data.iloc[:, 0].fillna(-1)
    pay_ts = data.iloc[:, 1].apply(extract_first_from_numeric_array)
    refund_ts = data.iloc[:, 2].apply(extract_first_from_numeric_array)
    click_ts = click_ts.apply(calc_seconds_since_start)
    pay_ts = pay_ts.apply(calc_seconds_since_start)
    refund_ts = refund_ts.apply(calc_seconds_since_start)

    feature_cols = [i for i in range(3, 25)]
    features_df = data.iloc[:,feature_cols].applymap(extract_first_from_array)

    features_df,current_mapping_dict = process_features_with_mapping_dict(features_df)

    final_mapping_dict = current_mapping_dict

    if mapping_output_path is not None:
        if os.path.exists(mapping_output_path):
            print(f"Loading existing mapping dictionary from {mapping_output_path}")
            with open(mapping_output_path, 'rb') as f:
                saved_mapping_dict = pickle.load(f)
            
            for col in current_mapping_dict.keys():
                if col not in saved_mapping_dict:
                    saved_mapping_dict[col] = {}
                for raw_val, hash_id in current_mapping_dict[col].items():
                    if raw_val not in saved_mapping_dict[col]:
                        saved_mapping_dict[col][raw_val] = hash_id
            final_mapping_dict = saved_mapping_dict
            print("Mapping dictionary updated with new values.")
        else:
            print(f"No existing mapping file found at {mapping_output_path}, will create a new one.")

        with open(mapping_output_path, 'wb') as f:
            pickle.dump(final_mapping_dict, f)
        print(f"Updated mapping dictionary saved to {mapping_output_path}")


    merged_df = pd.concat([click_ts,pay_ts,refund_ts,features_df], axis=1)
    merged_df.to_csv(output_path, sep="\t", index=False,header=False,mode='a')
    print(merged_df.head())
    print(f"Processed data written to {output_path}")
    return merged_df

def batch_preprocess_data(input_dir, output_path,mapping_features_output_file):

    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    for filename in tqdm(txt_files):
        data_path = os.path.join(input_dir, filename)
        preprocess_data_df(data_path, output_path,mapping_features_output_file)


if __name__ == "__main__":
    input_directory = "/mnt/nfs/luomingxuan/air_data_20250501_20250525_encrypted_source/air_data/"
    output_file = "./processed_encrypted_data_expand_with_MappingDict.txt"
    mapping_features_output_file = "./processed_encrypted_data_expand_with_MappingDict_feature_mapping.pkl"
    batch_preprocess_data(input_directory, output_file,mapping_features_output_file)
