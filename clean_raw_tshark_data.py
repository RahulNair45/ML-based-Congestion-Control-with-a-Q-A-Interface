import pandas as pd

# How to convert a pcapng file to csv with tshark
# Example:
# & "C:\Program Files\Wireshark\tshark.exe" -r "File path" -T fields -e frame.time_epoch -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e tcp.stream -e udp.stream -e frame.len -e tcp.len -e udp.length -e ip.proto -e tcp.flags -e tcp.window_size_value -e tcp.analysis.retransmission -e tcp.analysis.duplicate_ack -e tcp.analysis.lost_segment -e tcp.analysis.bytes_in_flight -e tcp.analysis.ack_rtt -e dns.qry.name -e tls.handshake.extensions_server_name -E "header=y" -E "separator=," -E "quote=d" -E "occurrence=f" > "output.csv"

input_file = "specific_captures/extent_congestion_every_30s_5__10__15__pLoss.pcapng/extent_congestion_every_30s_5__10__15__pLoss.csv"
output_file = "specific_captures/extent_congestion_every_30s_5__10__15__pLoss.pcapng/pLoss_clean.csv"

# congestion labeling
# set to 0 or 1 if the whole capture has one congestion label
# use None if only certain time ranges should be marked congested
whole_capture_congestion_label = None
# examples:
# whole_capture_congestion_label = 0
# whole_capture_congestion_label = 1
# whole_capture_congestion_label = None

# define congested time ranges in seconds from the start of the capture
# only used when whole_capture_congestion_label = None
congestion_ranges_relative = [
    # (start_second, end_second),
    # (0, 6),
    # (180, 413)
    (32, 63),
    (92, 122),
    (154, 205)
]

# whole-capture traffic labels
# set any of these to 1 if the whole capture contains that traffic type
# use all 0s if you want to label by time ranges instead
whole_capture_traffic_labels = {
    "traffic_browsing": 1,
    "traffic_email_chat": 0,
    "traffic_streaming": 1,
    "traffic_file_transfer": 0,
    "traffic_voice_video_call": 0,
}

# set this to True if you want to use the whole-capture traffic labels above
# set to False if you want to use time ranges below
use_whole_capture_traffic_labels = True

# define traffic type ranges in seconds from the start of the capture
# each tuple is (start_second, end_second, [list of active traffic labels])
# only used when use_whole_capture_traffic_labels = False
traffic_ranges_relative = [
    # (0, 38, ["traffic_browsing"]),
    # (38, 81, ["traffic_browsing", "traffic_email_chat"]),
    # (81, 117, ["traffic_browsing", "traffic_email_chat", "traffic_streaming"]),
    # (117, 151, ["traffic_browsing", "traffic_email_chat", "traffic_streaming", "traffic_voice_video_call"]),
    # (151, 241, ["traffic_browsing", "traffic_email_chat", "traffic_streaming", "traffic_voice_video_call", "traffic_file_transfer"]),
    # (241, 271, ["traffic_browsing", "traffic_email_chat", "traffic_streaming", "traffic_voice_video_call"]),
    # (271, 299, ["traffic_browsing", "traffic_email_chat", "traffic_streaming"]),
    # (299, 330, ["traffic_browsing", "traffic_email_chat"]),
    # (330, 413, ["traffic_browsing"])

]

# load the raw tshark csv
df = pd.read_csv(input_file, encoding="utf-16")
print(df.head())

# replace blank strings with missing values
df = df.replace("", pd.NA)

# convert columns that should be numeric
num_cols = [
    "frame.time_epoch",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
    "tcp.stream",
    "udp.stream",
    "frame.len",
    "tcp.len",
    "udp.length",
    "ip.proto",
    "tcp.window_size_value",
    "tcp.analysis.bytes_in_flight",
    "tcp.analysis.ack_rtt"
]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# combine TCP and UDP ports into one source/destination port
df["src_port"] = df["tcp.srcport"].fillna(df["udp.srcport"])
df["dst_port"] = df["tcp.dstport"].fillna(df["udp.dstport"])

# combine TCP and UDP stream IDs into one stream column
df["stream_id"] = df["tcp.stream"].fillna(df["udp.stream"])

# create simple protocol flags
df["is_tcp"] = (df["ip.proto"] == 6).astype(int)
df["is_udp"] = (df["ip.proto"] == 17).astype(int)

# turn tshark analysis columns into 0/1 flags
for col in [
    "tcp.analysis.retransmission",
    "tcp.analysis.duplicate_ack",
    "tcp.analysis.lost_segment"
]:
    if col in df.columns:
        df[col] = df[col].notna().astype(int)

# drop rows missing the basic packet info
df = df.dropna(subset=["frame.time_epoch", "ip.src", "ip.dst", "frame.len"])

# remove columns that are completely empty
df = df.dropna(axis=1, how="all")

# create relative time in seconds from the start of the capture
df["relative_time_sec"] = df["frame.time_epoch"] - df["frame.time_epoch"].min()

# assign congestion label
if whole_capture_congestion_label is not None:
    df["congestion_label"] = whole_capture_congestion_label
else:
    df["congestion_label"] = 0
    for start_sec, end_sec in congestion_ranges_relative:
        df.loc[
            (df["relative_time_sec"] >= start_sec) &
            (df["relative_time_sec"] <= end_sec),
            "congestion_label"
        ] = 1

# initialize traffic label columns
traffic_cols = [
    "traffic_browsing",
    "traffic_email_chat",
    "traffic_streaming",
    "traffic_file_transfer",
    "traffic_voice_video_call"
]

for col in traffic_cols:
    df[col] = 0

# assign traffic labels
if use_whole_capture_traffic_labels:
    for col, value in whole_capture_traffic_labels.items():
        if col in df.columns:
            df[col] = value
else:
    for start_sec, end_sec, active_traffic_list in traffic_ranges_relative:
        mask = (
            (df["relative_time_sec"] >= start_sec) &
            (df["relative_time_sec"] < end_sec)
        )
        for traffic_col in active_traffic_list:
            if traffic_col in df.columns:
                df.loc[mask, traffic_col] = 1

# keep the columns that are most useful later
keep_cols = [
    "frame.time_epoch",
    "relative_time_sec",
    "congestion_label",
    "traffic_browsing",
    "traffic_email_chat",
    "traffic_streaming",
    "traffic_file_transfer",
    "traffic_voice_video_call",
    "ip.src",
    "ip.dst",
    "src_port",
    "dst_port",
    "stream_id",
    "frame.len",
    "tcp.len",
    "udp.length",
    "ip.proto",
    "is_tcp",
    "is_udp",
    "tcp.flags",
    "tcp.window_size_value",
    "tcp.analysis.retransmission",
    "tcp.analysis.duplicate_ack",
    "tcp.analysis.lost_segment",
    "tcp.analysis.bytes_in_flight",
    "tcp.analysis.ack_rtt",
    "dns.qry.name",
    "tls.handshake.extensions_server_name"
]

# only keep columns that still exist
keep_cols = [col for col in keep_cols if col in df.columns]
df_clean = df[keep_cols].copy()

# save cleaned csv
df_clean.to_csv(output_file, index=False)

print(df_clean.head())
print("\nColumns kept:")
print(df_clean.columns.tolist())
print("\nCongestion label counts:")
print(df_clean["congestion_label"].value_counts())

for col in traffic_cols:
    print(f"\n{col} counts:")
    print(df_clean[col].value_counts())

print(f"\nSaved cleaned file to {output_file}")