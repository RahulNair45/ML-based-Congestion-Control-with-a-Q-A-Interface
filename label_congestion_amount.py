import pandas as pd

input_file = "specific_captures/extent_congestion_every_30s_5__10__15__pLoss.pcapng/pLoss_clean.csv"
output_file = "pLoss_clean_loss_labels.csv"

# define packet-loss ranges in seconds from the start of the capture
# format: (start_second, end_second, packet_loss_level)
packet_loss_ranges_relative = [
    # (start, end, loss level)
    # Example:
    # (30, 60, 0.05),    # 5% packet loss
    # (90, 120, 0.10),   # 10% packet loss
    # (150, 180, 0.20),  # 20% packet loss
    (32, 63, 0.05),
    (92, 122, 0.10),
    (154, 205, 0.20)
]


# load the cleaned packet-level CSV
df = pd.read_csv(input_file)

# make sure relative time is numeric
df["relative_time_sec"] = pd.to_numeric(df["relative_time_sec"], errors="coerce")

# start every packet as no induced packet loss
df["packet_loss_level"] = 0.00

# label packet loss level based on relative time ranges
for start_sec, end_sec, loss_level in packet_loss_ranges_relative:
    df.loc[
        (df["relative_time_sec"] >= start_sec) &
        (df["relative_time_sec"] <= end_sec),
        "packet_loss_level"
    ] = loss_level

# if packet loss is greater than 0, mark it as congested too
df.loc[df["packet_loss_level"] > 0, "congestion_label"] = 1

# save the updated cleaned CSV
df.to_csv(output_file, index=False)

print(df.head())
print("\nPacket loss level counts:")
print(df["packet_loss_level"].value_counts().sort_index())

print("\nCongestion label counts:")
print(df["congestion_label"].value_counts())

print(f"\nSaved labeled file to {output_file}")