import pandas as pd
import numpy as np

# input_file = "combined_dataset.csv"
# output_file = "aggregated_dataset_5.csv"
# packet_output_file = "packet_with_windows_5.csv"

input_file = "pLoss_clean_loss_labels.csv"
output_file = "pLoss_aggregated_5.csv"
packet_output_file = "pLoss_packet_windows_5.csv"

WINDOW_SIZE = 5   # size of time window in seconds 

df = pd.read_csv(input_file)

# add source file name if this is a single cleaned capture instead of combined data
if "source_file" not in df.columns:
    df["source_file"] = input_file

# add packet loss label for older files that do not have this column
# older congested captures are treated as 5% loss, non-congested as 0%
if "packet_loss_level" not in df.columns:
    df["packet_loss_level"] = np.where(df["congestion_label"] == 1, 0.05, 0.0)

# add missing traffic columns as 0 so the aggregation does not crash
traffic_cols = [
    "traffic_browsing",
    "traffic_email_chat",
    "traffic_streaming",
    "traffic_file_transfer",
    "traffic_voice_video_call"
]

for col in traffic_cols:
    if col not in df.columns:
        df[col] = 0

# Convert important columns to numeric
num_cols = [
    "frame.time_epoch",
    "relative_time_sec",
    "frame.len",
    "stream_id",
    "tcp.window_size_value",
    "tcp.analysis.retransmission",
    "tcp.analysis.duplicate_ack",
    "tcp.analysis.lost_segment",
    "tcp.analysis.bytes_in_flight",
    "tcp.analysis.ack_rtt",
    "is_tcp",
    "is_udp",
    "congestion_label",
    "packet_loss_level",
    "traffic_browsing",
    "traffic_email_chat",
    "traffic_streaming",
    "traffic_file_transfer",
    "traffic_voice_video_call"
]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# fill missing packet loss values as no packet loss
df["packet_loss_level"] = df["packet_loss_level"].fillna(0.0)

# fill missing label values safely
df["congestion_label"] = df["congestion_label"].fillna(0)

for col in traffic_cols:
    df[col] = df[col].fillna(0)

# drop invalid rows
df = df.dropna(subset=["relative_time_sec", "frame.len"])

# Create time windows

# assigns each packet to a window based on relative time
df["time_window"] = (df["relative_time_sec"] // WINDOW_SIZE).astype(int)

# creates a clean sequential window index per capture file
df["window_index"] = df.groupby("source_file")["time_window"].rank(method="dense").astype(int) - 1

# save packet-level data with window labels
df.to_csv(packet_output_file, index=False)


# Aggregate features per window
features = df.groupby(["source_file", "time_window"]).agg(

    # traffic volume 
    total_packets=("frame.len", "count"),          # total number of packets in window
    total_bytes=("frame.len", "sum"),              # total bytes transferred
    mean_packet_size=("frame.len", "mean"),        # average packet size
    max_packet_size=("frame.len", "max"),          # largest packet
    std_packet_size=("frame.len", "std"),          # variation in packet sizes

    # flow behavior
    active_flows=("stream_id", "nunique"),         # number of unique flows (connections)

    # congestion signals 
    retransmissions=("tcp.analysis.retransmission", "sum"),  # number of retransmitted packets
    duplicate_acks=("tcp.analysis.duplicate_ack", "sum"),    # number of duplicate ACKs
    lost_segments=("tcp.analysis.lost_segment", "sum"),      # number of detected packet losses

    # delay / latency 
    mean_rtt=("tcp.analysis.ack_rtt", "mean"),     # average round-trip time
    max_rtt=("tcp.analysis.ack_rtt", "max"),       # maximum RTT observed
    std_rtt=("tcp.analysis.ack_rtt", "std"),       # RTT variation

    # network load / queue 
    mean_bytes_in_flight=("tcp.analysis.bytes_in_flight", "mean"),  # avg unacknowledged data
    max_bytes_in_flight=("tcp.analysis.bytes_in_flight", "max"),    # peak in-flight data

    # TCP behavior 
    mean_window_size=("tcp.window_size_value", "mean"),  # avg TCP receive window size

    # protocol mix 
    tcp_packet_count=("is_tcp", "sum"),           # number of TCP packets
    udp_packet_count=("is_udp", "sum"),           # number of UDP packets

    # labels 
    # congestion: majority vote across packets in window
    congestion_label=("congestion_label", lambda x: int(x.mean() >= 0.5)),

    # packet loss: use highest packet loss level present in the window
    packet_loss_level=("packet_loss_level", "max"),

    # traffic types: if ANY packet has it → mark as present
    traffic_browsing=("traffic_browsing", "max"),
    traffic_email_chat=("traffic_email_chat", "max"),
    traffic_streaming=("traffic_streaming", "max"),
    traffic_file_transfer=("traffic_file_transfer", "max"),
    traffic_voice_video_call=("traffic_voice_video_call", "max")

).reset_index()

# Derived features
# normalize traffic by time window
features["packets_per_sec"] = features["total_packets"] / WINDOW_SIZE   # packet rate
features["bytes_per_sec"] = features["total_bytes"] / WINDOW_SIZE       # throughput

# protocol ratios
features["tcp_ratio"] = features["tcp_packet_count"] / features["total_packets"]  # fraction of TCP traffic
features["udp_ratio"] = features["udp_packet_count"] / features["total_packets"]  # fraction of UDP traffic

# per-flow averages
features["avg_bytes_per_flow"] = features["total_bytes"] / features["active_flows"].replace(0, np.nan)   # avg bytes per flow
features["avg_packets_per_flow"] = features["total_packets"] / features["active_flows"].replace(0, np.nan)  # avg packets per flow

# Handle missing values
fill_zero_cols = [
    "mean_rtt", "max_rtt", "std_rtt",
    "mean_bytes_in_flight", "max_bytes_in_flight",
    "mean_window_size", "std_packet_size"
]

for col in fill_zero_cols:
    if col in features.columns:
        features[col] = features[col].fillna(0)

# handle divide-by-zero cases
features["avg_bytes_per_flow"] = features["avg_bytes_per_flow"].fillna(0)
features["avg_packets_per_flow"] = features["avg_packets_per_flow"].fillna(0)

# keep packet loss label clean
features["packet_loss_level"] = features["packet_loss_level"].fillna(0.0)

# Move labels to the end
label_cols = [
    "congestion_label",
    "packet_loss_level",
    "traffic_browsing",
    "traffic_email_chat",
    "traffic_streaming",
    "traffic_file_transfer",
    "traffic_voice_video_call"
]

feature_cols = [col for col in features.columns if col not in label_cols]
features = features[feature_cols + label_cols]

features.to_csv(output_file, index=False)

print("\nSaved packet-level windows to:", packet_output_file)
print("Saved aggregated dataset to:", output_file)

print("\nPacket loss level distribution:")
print(features["packet_loss_level"].value_counts().sort_index())

print("\nCongestion label distribution:")
print(features["congestion_label"].value_counts())

print("\nSample output:")
print(features.head())