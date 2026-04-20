import pandas as pd
import numpy as np
from collections import defaultdict  #can handle missing keys (useufl for our data)

#data loading and cleaning
def make_flows(csv_file, label):
    df = pd.read_csv(csv_file)  #creating dataframe from csv file
    df['Time']   = pd.to_numeric(df['Time'],   errors='coerce')   #converting time and length columns to numbers, non-numeric values to NaN
    df['Length'] = pd.to_numeric(df['Length'], errors='coerce')
    df = df.dropna(subset=['Time','Length','Source','Destination'])   #deletes rows without important info

    #grouping packets into flows
    flows = defaultdict(list)    #dictionary

    for _, row in df.iterrows():
        src, dst = str(row['Source']), str(row['Destination'])   #source and desitnation addresses
        key = (min(src,dst), max(src,dst), str(row['Protocol']))      #bi-directional flow key (both min and max with both src and dst)
        flows[key].append((row['Time'], row['Length']))        #adds the timestamp and length to that flow's list

    #feature engineering
    rows = []
    for key, packets in flows.items():
        if len(packets) < 4:        #to exlcude short conversations with less than 4 packets
            continue

        times  = [p[0] for p in packets]        #extracting all timestamps in this flow
        sizes  = [p[1] for p in packets]        #extracting all packet sizes in this flow
        iats   = [times[i+1]-times[i] for i in range(len(times)-1)]   #calculating IAT (inter arrival times (gap b/w consecutive packets))

        if not iats:
            continue

        duration = times[-1] - times[0]        #total duration of flow
        if duration <= 0:
            continue

        #dataset columns
        rows.append({
            'duration':           duration,
            'mean_flowiat':       np.mean(iats),   #time stats
            'std_flowiat':        np.std(iats),
            'min_flowiat':        np.min(iats),
            'max_flowiat':        np.max(iats),
            'flowPktsPerSecond':  len(times) / duration,  #throughput
            'flowBytesPerSecond': sum(sizes) / duration, 
            'mean_pkt_size':      np.mean(sizes),   #packet stats
            'min_pkt_size':       np.min(sizes),
            'max_pkt_size':       np.max(sizes),
            'src':                key[0],     #details
            'dst':                key[1],
            'protocol':           key[2],
            'congested':          label        #1=congested, 0=normal
        })

    return pd.DataFrame(rows)


# file saving
print("Congested capture.")
congested = make_flows('local_voip_capture.csv', label=1)
print(f"  {len(congested)} flows")

print("Normal capture.")
normal = make_flows('void_capture_no_congestion.csv', label=0)
print(f"  {len(normal)} flows")

#one large dataset
dataset = pd.concat([congested, normal], ignore_index=True)
dataset.to_csv('congestion_dataset.csv', index=False)

print(f"\nSaved congestion_dataset.csv")
print(f"Total flows : {len(dataset)}")
print(f"Congested   : {dataset['congested'].sum()}")
print(f"Normal      : {(dataset['congested']==0).sum()}")