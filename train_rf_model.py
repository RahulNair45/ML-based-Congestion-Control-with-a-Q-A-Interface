import pandas as pd
import pickle      #to save a trained model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#for the model
FEATURES = [
    'duration',
    'mean_flowiat', 'std_flowiat', 'min_flowiat', 'max_flowiat',
    'flowPktsPerSecond', 'flowBytesPerSecond',
    'mean_pkt_size', 'min_pkt_size', 'max_pkt_size',
]

# extracting data and filling NaN with 0
df = pd.read_csv('congestion_dataset.csv')
X  = df[FEATURES].fillna(0)
#what it trains on
y  = df['congested']

#splitting into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# training a random forest classifier
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# evaluating the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Normal','Congested']))

# saving the model
with open('congestion_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'features': FEATURES}, f)

print("Saved congestion_model.pkl")