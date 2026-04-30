import pandas as pd
import numpy as np
import joblib

# Run cong_classification_models.py before this
# which saves:
#   - best_congestion_model_5_30.pkl      (the best trained model)
#   - X_5_30_test.csv                     (test feature rows)
#   - y_5_30_test.csv                     (true test labels)
#   - test_5_30_full_context.csv          (test rows with traffic type columns)

# ── swap these filenames if using a different window size or test split─────────
MODEL_FILE        = "best_congestion_model_5_30.pkl"
X_TEST_FILE       = "X_5_30_test.csv"
Y_TEST_FILE       = "y_5_30_test.csv"
CONTEXT_FILE      = "test_5_30_full_context.csv"


# Load model + test data

def load_everything():
    print(f"Loading model from {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)

    print(f"Loading test data...")
    X_test  = pd.read_csv(X_TEST_FILE)
    y_test  = pd.read_csv(Y_TEST_FILE).values.ravel()
    context = pd.read_csv(CONTEXT_FILE)

    return model, X_test, y_test, context


# Build the network state summary
#
# Runs the model on the test set and computes averages across all windows
# to produce a single dictionary representing "what the network looked like"
# during this capture. The Q&A rules read from this dictionary.

# traffic type column names as they appear in the context CSV
TRAFFIC_COLS = [
    "traffic_browsing",
    "traffic_email_chat",
    "traffic_streaming",
    "traffic_file_transfer",
    "traffic_voice_video_call",
]

# human-readable names for each traffic column
TRAFFIC_LABELS = {
    "traffic_browsing":          "Browsing",
    "traffic_email_chat":        "Email/Chat",
    "traffic_streaming":         "Streaming",
    "traffic_file_transfer":     "File Transfer",
    "traffic_voice_video_call":  "Voice/Video Call",
}

# which traffic types are latency-sensitive
LATENCY_SENSITIVE = {"traffic_voice_video_call", "traffic_email_chat"}

# which traffic types are bulk / high-bandwidth
BULK_TYPES        = {"traffic_streaming", "traffic_file_transfer"}


def build_network_state(model, X_test, y_test, context):
    # run model on all test windows
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of congestion

    # overall congestion: majority vote across all test windows
    congested    = bool(y_pred.mean() >= 0.5)
    confidence   = float(y_proba.mean())

    #per-window feature averages 
    def col_mean(name):
        """Return mean of a column if it exists, else None."""
        return float(X_test[name].mean()) if name in X_test.columns else None

    mean_rtt         = col_mean("mean_rtt")
    mean_bps         = col_mean("bytes_per_sec")
    retrans_total    = float(X_test["retransmissions"].sum()) if "retransmissions" in X_test.columns else 0
    total_packets    = float(X_test["total_packets"].sum())   if "total_packets"   in X_test.columns else None
    dup_acks         = float(X_test["duplicate_acks"].sum())  if "duplicate_acks"  in X_test.columns else 0
    lost_segs        = float(X_test["lost_segments"].sum())   if "lost_segments"   in X_test.columns else 0
    active_flows_avg = col_mean("active_flows")
    mean_bytes_inflight = col_mean("mean_bytes_in_flight")
    mean_window_size = col_mean("mean_window_size")

    retrans_rate = (retrans_total / total_packets) if total_packets and total_packets > 0 else 0.0

    # traffic type breakdown from context columns 
    traffic_counts = {}  # label → count of windows where that type was active

    for col in TRAFFIC_COLS:
        if col in context.columns:
            count = int(context[col].sum())
            if count > 0:
                label = TRAFFIC_LABELS[col]
                traffic_counts[label] = count

    total_windows = len(context)

    # dominant type = the one present in the most windows
    dominant_col  = max(
        (col for col in TRAFFIC_COLS if col in context.columns),
        key=lambda c: context[c].sum() if c in context.columns else 0,
        default=None
    )
    dominant_type = TRAFFIC_LABELS[dominant_col] if dominant_col else "Unknown"

    # fraction of windows with latency-sensitive or bulk traffic active
    def type_fraction(type_set):
        active = context[[c for c in type_set if c in context.columns]]
        if active.empty:
            return 0.0
        return float((active.max(axis=1) == 1).mean())

    latency_fraction = type_fraction(LATENCY_SENSITIVE)
    bulk_fraction    = type_fraction(BULK_TYPES)

    #  per-traffic-type congestion accuracy (from y_test vs y_pred) 
    context_eval = context.copy()
    context_eval["y_true"] = y_test[:len(context_eval)]
    context_eval["y_pred"] = y_pred[:len(context_eval)]

    traffic_f1 = {}
    for col in TRAFFIC_COLS:
        if col not in context_eval.columns:
            continue
        subset = context_eval[context_eval[col] == 1]
        if len(subset) == 0:
            continue
        from sklearn.metrics import f1_score
        f1 = f1_score(subset["y_true"], subset["y_pred"], zero_division=0)
        traffic_f1[TRAFFIC_LABELS[col]] = f1

    return {
        "congested":            congested,
        "confidence":           confidence,
        "mean_rtt_sec":         mean_rtt,
        "mean_bps":             mean_bps,
        "total_packets":        total_packets,
        "retrans_rate":         retrans_rate,
        "duplicate_acks":       dup_acks,
        "lost_segments":        lost_segs,
        "active_flows_avg":     active_flows_avg,
        "mean_bytes_inflight":  mean_bytes_inflight,
        "mean_window_size":     mean_window_size,
        "dominant_type":        dominant_type,
        "traffic_counts":       traffic_counts,
        "total_windows":        total_windows,
        "latency_fraction":     latency_fraction,
        "bulk_fraction":        bulk_fraction,
        "traffic_f1":           traffic_f1,
    }


#Formatting

def fmt_bps(bps):
    if bps is None: return "unknown"
    if bps >= 1_000_000: return f"{bps/1_000_000:.1f} MB/s"
    if bps >= 1_000:     return f"{bps/1_000:.1f} KB/s"
    return f"{bps:.0f} B/s"

def fmt_rtt(sec):
    if sec is None: return "unknown"
    return f"{sec * 1000:.1f} ms"

def fmt_pct(frac):
    return f"{frac * 100:.1f}%"

def fmt_conf(frac):
    return f"{frac * 100:.0f}%"

def traffic_breakdown_str(counts):
    if not counts: return "no traffic data"
    parts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return ", ".join(f"{label}: {n} windows" for label, n in parts)


#Keyword matching

KEYWORDS = {
    "lag":        ["lag", "lagg", "lagging", "slow", "delay", "jitter",
                   "freeze", "freezing", "stutter", "buffering", "choppy"],
    "call":       ["call", "zoom", "teams", "meet", "voip", "voice",
                   "video call", "conference", "skype"],
    "congestion": ["congest", "congestion", "bottleneck", "saturated",
                   "overloaded", "bandwidth full", "maxed out"],
    "bandwidth":  ["bandwidth", "speed", "throughput", "how fast",
                   "how much bandwidth", "using the most", "consuming"],
    "traffic":    ["traffic", "what type", "application", "what is running",
                   "what app", "what service", "which app"],
    "health":     ["health", "healthy", "status", "overview", "summary",
                   "how is", "how's", "state of", "report", "overall"],
    "retransmit": ["retransmit", "retransmission", "packet loss",
                   "lost packet", "dropped packet", "lost segment"],
    "rtt":        ["rtt", "round trip", "ping", "latency", "response time",
                   "ms", "milliseconds"],
    "flows":      ["flow", "stream", "connection", "how many", "active"],
    "streaming":  ["streaming", "netflix", "youtube", "video stream", "watch"],
    "download":   ["download", "file transfer", "uploading", "upload"],
    "accuracy":   ["accurate", "accuracy", "how well", "performance",
                   "model", "correct", "score", "f1", "precision", "recall"],
}

def classify_question(question):
    q = question.lower()
    return [cat for cat, words in KEYWORDS.items() if any(w in q for w in words)]


# Rule-based answer engine

def answer(question, state):
    cats = classify_question(question)

    # convenience unpacking
    congested    = state["congested"]
    conf         = state["confidence"]
    dom          = state["dominant_type"]
    rtt          = state["mean_rtt_sec"]
    bps          = state["mean_bps"]
    ret_rate     = state["retrans_rate"]
    lat_frac     = state["latency_fraction"]
    bulk_frac    = state["bulk_fraction"]
    counts       = state["traffic_counts"]
    dup_acks     = state["duplicate_acks"]
    lost_segs    = state["lost_segments"]
    flows        = state["active_flows_avg"]
    traffic_f1   = state["traffic_f1"]

    # model accuracy / performance questions 
    if "accuracy" in cats:
        lines = ["Model performance on the test set:"]
        if traffic_f1:
            lines.append("\nCongestion detection F1 by traffic type:")
            for ttype, f1 in sorted(traffic_f1.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {ttype}: {fmt_pct(f1)}")
        lines.append(
            "\nNote: F1 scores below reflect how accurately congestion was detected "
            "when each traffic type was active. File Transfer and Streaming windows "
            "are easiest to classify; Voice/Video Call windows are hardest because "
            "VoIP produces weaker congestion signals (small UDP packets, low retransmission rate)."
        )
        return "\n".join(lines)

    # call quality / lag questions 
    if "lag" in cats or "call" in cats:
        if congested:
            msg = f"Congestion was detected in this capture (model confidence: {fmt_conf(conf)}).\n"
            if bulk_frac > 0.3:
                msg += (
                    f"A significant portion of windows ({fmt_pct(bulk_frac)}) had bulk traffic "
                    f"(Streaming or File Transfer) active. This type of traffic competes with "
                    f"latency-sensitive applications like video calls for bandwidth.\n"
                )
            if rtt is not None:
                msg += f"Average round-trip time: {fmt_rtt(rtt)}. "
            if ret_rate > 0.01:
                msg += (
                    f"Retransmission rate: {fmt_pct(ret_rate)} — packets are being "
                    f"dropped and resent, which directly causes jitter in VoIP calls.\n"
                )
            msg += (
                "\nRecommendation: Pause or reschedule large downloads and streaming "
                "sessions during calls to reduce competition on the link."
            )
        else:
            msg = (
                f"No congestion was detected in this capture (confidence: {fmt_conf(1 - conf)}).\n"
                "If you experienced lag, the cause is likely external — such as high latency "
                "on the path to the remote server rather than congestion on your local link."
            )
        return msg

    # direct congestion query 
    if "congestion" in cats:
        if congested:
            msg = f"Yes — congestion was detected (confidence: {fmt_conf(conf)}).\n"
            msg += f"Dominant traffic type: {dom}. "
            if rtt is not None:
                msg += f"Avg RTT: {fmt_rtt(rtt)}. "
            if ret_rate > 0:
                msg += f"Retransmission rate: {fmt_pct(ret_rate)}."
        else:
            msg = (
                f"No congestion was detected (confidence: {fmt_conf(1 - conf)}).\n"
                "The network appeared to be operating normally during this capture."
            )
        return msg

    # bandwidth / throughput questions 
    if "bandwidth" in cats or "download" in cats or "streaming" in cats:
        msg = f"Dominant traffic type: {dom}.\n"
        msg += f"Traffic breakdown: {traffic_breakdown_str(counts)}.\n"
        if bps is not None:
            msg += f"Average throughput: {fmt_bps(bps)}.\n"
        if bulk_frac > 0.4:
            msg += (
                f"Bulk transfer traffic (Streaming, File Transfer) was active in "
                f"{fmt_pct(bulk_frac)} of windows — the likely dominant source of "
                f"bandwidth consumption."
            )
        return msg

    # traffic type questions
    if "traffic" in cats:
        msg = f"Traffic breakdown for this capture:\n{traffic_breakdown_str(counts)}\n\n"
        msg += f"Dominant type: {dom}.\n"
        if lat_frac > 0:
            msg += f"Latency-sensitive traffic (VoIP, Email/Chat): active in {fmt_pct(lat_frac)} of windows.\n"
        if bulk_frac > 0:
            msg += f"Bulk transfer traffic (Streaming, File Transfer): active in {fmt_pct(bulk_frac)} of windows."
        return msg

    # RTT / latency questions 
    if "rtt" in cats:
        if rtt is not None:
            msg = f"Average TCP round-trip time: {fmt_rtt(rtt)}.\n"
            if rtt > 0.1:
                msg += "This is high (above 100 ms) and may indicate congestion or a distant server."
            elif rtt > 0.05:
                msg += "This is moderate — acceptable for most applications but noticeable for gaming or VoIP."
            else:
                msg += "This is low, which is good for latency-sensitive applications."
        else:
            msg = "RTT data was not available in this capture."
        return msg

    # retransmission / packet loss questions
    if "retransmit" in cats:
        msg = f"Retransmission rate: {fmt_pct(ret_rate)}.\n"
        if dup_acks > 0:
            msg += f"Duplicate ACKs: {int(dup_acks)}.\n"
        if lost_segs > 0:
            msg += f"Lost segments: {int(lost_segs)}.\n"
        if ret_rate > 0.05:
            msg += (
                "A rate above 5% is a strong indicator of congestion or link issues. "
                "Many packets are being dropped and resent, degrading throughput and latency."
            )
        elif ret_rate > 0.01:
            msg += "A rate between 1–5% suggests some packet loss — noticeable for TCP applications."
        else:
            msg += "The retransmission rate is low, indicating a relatively clean link."
        return msg

    #  flow count questions
    if "flows" in cats:
        msg = f"Total classified windows: {state['total_windows']}.\n"
        if flows is not None:
            msg += f"Average active flows per window: {flows:.1f}.\n"
        msg += f"Traffic breakdown: {traffic_breakdown_str(counts)}."
        return msg

    # health / summary (default) 
    status = "CONGESTED" if congested else "HEALTHY"
    lines = [
        f"=== Network State Summary ===",
        f"Status:          {status} (confidence: {fmt_conf(conf)})",
        f"Dominant traffic:{dom}",
    ]
    if bps is not None:
        lines.append(f"Avg throughput:  {fmt_bps(bps)}")
    if rtt is not None:
        lines.append(f"Avg RTT:         {fmt_rtt(rtt)}")
    lines.append(f"Retransmit rate: {fmt_pct(ret_rate)}")
    if flows is not None:
        lines.append(f"Avg active flows:{flows:.1f} per window")
    lines.append(f"\nTraffic breakdown:\n  {traffic_breakdown_str(counts)}")

    if congested:
        lines.append("\nAssessment: Congestion was detected in this capture.")
        if bulk_frac > 0.3:
            lines.append(
                f"Bulk traffic was active in {fmt_pct(bulk_frac)} of windows "
                f"and is likely contributing to link saturation."
            )
        if lat_frac > 0:
            lines.append(
                f"Latency-sensitive traffic was active in {fmt_pct(lat_frac)} of windows "
                f"and may have been impacted."
            )
    else:
        lines.append(
            "\nAssessment: No congestion detected. "
            "The network appeared to operate within normal bounds during this capture."
        )

    return "\n".join(lines)

def main():
    print("=" * 60)
    print("ML-based Network Q&A Interface")
    print("=" * 60)

    model, X_test, y_test, context = load_everything()

    print("Building network state from model predictions...")
    state = build_network_state(model, X_test, y_test, context)

    print("\n=== Loaded Network State ===")
    print(f"  Congested:        {state['congested']}  (confidence: {fmt_conf(state['confidence'])})")
    print(f"  Dominant traffic: {state['dominant_type']}")
    print(f"  Avg throughput:   {fmt_bps(state['mean_bps'])}")
    print(f"  Avg RTT:          {fmt_rtt(state['mean_rtt_sec'])}")
    print(f"  Retransmit rate:  {fmt_pct(state['retrans_rate'])}")
    print(f"  Traffic breakdown:{traffic_breakdown_str(state['traffic_counts'])}")
    if state["traffic_f1"]:
        print("\n  Congestion F1 by traffic type:")
        for t, f1 in sorted(state["traffic_f1"].items(), key=lambda x: x[1], reverse=True):
            print(f"    {t}: {fmt_pct(f1)}")

    print("\n" + "=" * 60)
    print("Ask a question about this network capture.")
    print("Type 'quit' to exit.\n")
    print("Example questions:")
    print("  - Why is my video call lagging?")
    print("  - Is there congestion?")
    print("  - What is using the most bandwidth?")
    print("  - What type of traffic is on the network?")
    print("  - What is the RTT?")
    print("  - Are there retransmissions?")
    print("  - How well does the model perform?")
    print("  - Give me a health summary.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            break

        response = answer(question, state)
        print(f"\nSystem: {response}\n")

main()