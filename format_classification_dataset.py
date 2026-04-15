from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff("VPN-nonVPN_dataset/Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff")
df = pd.DataFrame(data)

# decode byte strings into normal strings if needed
for col in df.select_dtypes([object]).columns:
    df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

df.to_csv("your_file.csv", index=False)
print(df.head())