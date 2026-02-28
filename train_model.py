import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------- LOAD DATA ----------------
data = pd.read_csv('D:\\prosperity_prognosticator\\startup data.csv')



print("\nDataset Shape:", data.shape)
print("\nMissing Values:\n", data.isnull().sum())

# ---------------- FEATURE SELECTION ----------------
cols = [
    'state_code',
    'funding_total_usd',
    'funding_rounds',
    'has_VC',
    'has_angel',
    'has_roundA',
    'has_roundB',
    'has_roundC',
    'has_roundD',
    'milestones',
    'relationships',
    'status'
]

data = data[cols]

# ---------------- PREPROCESSING ----------------

# Fill missing values
data.fillna(0, inplace=True)

# Encode state_code
le = LabelEncoder()
data['state_code'] = le.fit_transform(data['state_code'])

# Encode target variable
data['status'] = data['status'].map({'acquired':1,'closed':0})

print("\nStatus Distribution:\n", data['status'].value_counts())

# ---------------- SPLIT ----------------
X = data.drop('status',axis=1)
y = data['status']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train,y_train)

# ---------------- EVALUATION ----------------
train_acc = model.score(X_train,y_train)
test_acc = model.score(X_test,y_test)

print("\nTrain Accuracy:",train_acc)
print("Test Accuracy:",test_acc)

# ---------------- SAVE MODEL ----------------
joblib.dump(model,"random_forest_model.pkl")

print("\nModel Saved Successfully")