import pandas as pd
import glob
import os

# folder where all your cleaned CSVs are stored
input_folder = "cleaned_capture_data"
output_file = "combined_dataset.csv"

# find all csv files in the folder
file_paths = glob.glob(os.path.join(input_folder, "*.csv"))

all_dfs = []

for file in file_paths:
    print(f"Loading: {file}")
    
    df = pd.read_csv(file)

    # add column to track which capture it came from
    df["source_file"] = os.path.basename(file)

    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

combined_df.to_csv(output_file, index=False)

print("\nCombined dataset shape:", combined_df.shape)
print("Saved to:", output_file)

# DATASET STATS

print("\n===== DATASET DISTRIBUTIONS =====")

# Congestion distribution
if "congestion_label" in combined_df.columns:
    print("\n--- Congestion Label Distribution ---")
    print(combined_df["congestion_label"].value_counts())

# Traffic type distributions
traffic_cols = [
    "traffic_browsing",
    "traffic_email_chat",
    "traffic_streaming",
    "traffic_file_transfer",
    "traffic_voice_video_call"
]

print("\n--- Traffic Type Counts ---")
for col in traffic_cols:
    if col in combined_df.columns:
        print(f"{col}: {combined_df[col].sum()}")

# Source file distribution (how many rows per capture)
print("\n--- Samples per Capture File ---")
print(combined_df["source_file"].value_counts())

# congestion per file
if "congestion_label" in combined_df.columns:
    print("\n--- Congestion Distribution per File ---")
    print(pd.crosstab(combined_df["source_file"], combined_df["congestion_label"]))


# --- Congestion Label Distribution ---
# congestion_label
# 0    589620
# 1    230647
# Name: count, dtype: int64

# --- Traffic Type Counts ---
# traffic_browsing: 391088
# traffic_email_chat: 342058
# traffic_streaming: 294805
# traffic_file_transfer: 355754
# traffic_voice_video_call: 227500

# --- Samples per Capture File ---
# source_file
# mix_capture_clean.csv       318992
# file_download_clean.csv     217404
# gmail_chat_clean.csv        101440
# streaming_clean.csv          73277
# browsing_clean.csv           72096
# voice_vid_call_clean.csv     37058
# Name: count, dtype: int64

# --- Congestion Distribution per File ---
# congestion_label               0       1
# source_file                             
# browsing_clean.csv         56228   15868
# file_download_clean.csv   210539    6865
# gmail_chat_clean.csv       93468    7972
# mix_capture_clean.csv     172304  146688
# streaming_clean.csv        35830   37447
# voice_vid_call_clean.csv   21251   15807