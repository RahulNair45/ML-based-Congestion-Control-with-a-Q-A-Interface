import pandas as pd

# How to convert t-shark pcapng file to csv
# have wireshark downloaded and put in:
# & "C:\Program Files\Wireshark\tshark.exe" -r "File path" -T fields -e frame.time_epoch -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e tcp.stream -e udp.stream -e frame.len -e tcp.len -e udp.length -e ip.proto -e tcp.flags -e tcp.window_size_value -e tcp.analysis.retransmission -e tcp.analysis.duplicate_ack -e tcp.analysis.lost_segment -e tcp.analysis.bytes_in_flight -e tcp.analysis.ack_rtt -e dns.qry.name -e tls.handshake.extensions_server_name -E "header=y" -E "separator=," -E "quote=d" -E "occurrence=f" > "where you want your output\filename.csv"

# load the richer tshark csv
df = pd.read_csv("tshark_yes_congestion.csv", encoding="utf-16")
print(df.head())

# replace blank strings with missing values
df = df.replace("", pd.NA)

# convert columns that should be numbers into numeric form
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
# packet will usually have either TCP ports or UDP ports, not both
# gives us one cleaner port pair to work with later
df["src_port"] = df["tcp.srcport"].fillna(df["udp.srcport"])
df["dst_port"] = df["tcp.dstport"].fillna(df["udp.dstport"])

# combine TCP and UDP stream IDs into one stream column
# TCP and UDP flows are tracked in separate columns by tshark
# combining them makes it easier to group packets by one flow identifier
df["stream_id"] = df["tcp.stream"].fillna(df["udp.stream"])

# create simple protocol flags
df["is_tcp"] = (df["ip.proto"] == 6).astype(int)
df["is_udp"] = (df["ip.proto"] == 17).astype(int)

# turn tshark analysis columns into 0/1 flags
# tshark leaves these fields blank unless that event happened,
# converting them to 0/1 makes them easier to count and use in ML features
for col in [
    "tcp.analysis.retransmission",
    "tcp.analysis.duplicate_ack",
    "tcp.analysis.lost_segment"
]:
    if col in df.columns:
        df[col] = df[col].notna().astype(int)

# drop rows missing the basic packet info
df = df.dropna(subset=["frame.time_epoch", "ip.src", "ip.dst", "frame.len"])

# remove columns that are completely empty in this file
df = df.dropna(axis=1, how="all")

# keep the columns that are most useful for later feature building
keep_cols = [
    "frame.time_epoch",
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

# only keep columns that still exist after cleaning
keep_cols = [col for col in keep_cols if col in df.columns]
df_clean = df[keep_cols].copy()

# save the cleaned packet csv
df_clean.to_csv("tshark_yes_congestion_cleaned.csv", index=False)

print(df_clean.head())
print("\nColumns kept:")
print(df_clean.columns.tolist())
print("\nSaved as 'original_name'_cleaned.csv")