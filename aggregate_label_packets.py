import pandas as pd

input_file = "tshark_no_congestion_cleaned.csv"
output_file = "agg_non_cong_packets.csv"
label_value = 0   # 1 = congested, 0 = non-congested

df = pd.read_csv(input_file)

# make sure important columns are numeric
num_cols = [
    "frame.time_epoch",
    "frame.len",
    "tcp.analysis.retransmission",
    "tcp.analysis.duplicate_ack",
    "tcp.analysis.lost_segment",
    "tcp.analysis.bytes_in_flight",
    "tcp.analysis.ack_rtt",
    "is_tcp",
    "is_udp",
    "stream_id"
]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# drop rows missing the basic packet info
df = df.dropna(subset=["frame.time_epoch", "frame.len"])

# create 1-second time windows
# each packet gets assigned to a whole-second bucket
df["time_window"] = df["frame.time_epoch"].astype(int)

# group packets by time window and compute summary features
features = df.groupby("time_window").agg(
    total_packets=("frame.len", "count"),                 # number of packets in the window
    total_bytes=("frame.len", "sum"),                     # total traffic volume in bytes
    mean_packet_size=("frame.len", "mean"),               # average packet size
    active_flows=("stream_id", "nunique"),                # number of unique flows active
    retransmissions=("tcp.analysis.retransmission", "sum"),   # count retransmissions
    duplicate_acks=("tcp.analysis.duplicate_ack", "sum"),     # count duplicate ACKs
    lost_segments=("tcp.analysis.lost_segment", "sum"),       # count lost segment events
    mean_rtt=("tcp.analysis.ack_rtt", "mean"),            # average RTT seen in the window
    mean_bytes_in_flight=("tcp.analysis.bytes_in_flight", "mean"),  # average outstanding TCP data
    tcp_packet_count=("is_tcp", "sum"),                   # how many packets were TCP
    udp_packet_count=("is_udp", "sum")                    # how many packets were UDP
).reset_index()

# since the window is 1 second, packets/sec and bytes/sec are just totals
features["packets_per_sec"] = features["total_packets"]
features["bytes_per_sec"] = features["total_bytes"]

# fill missing values that happen when a window has no RTT or bytes-in-flight info
features["mean_rtt"] = features["mean_rtt"].fillna(0)
features["mean_bytes_in_flight"] = features["mean_bytes_in_flight"].fillna(0)


features["label"] = label_value

features.to_csv(output_file, index=False)

print(features.head())
print(f"Saved features to {output_file}")