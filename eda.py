import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('D:\\prosperity_prognosticator\\startup data.csv')


# Keep important columns
cols = [
    'funding_total_usd',
    'funding_rounds',
    'milestones',
    'relationships',
    'has_VC',
    'status'
]

data = data[cols]

# Fill missing
data.fillna(0, inplace=True)

# Encode status
data['status'] = data['status'].map({'acquired':1,'closed':0})

# ---------------- STATUS DISTRIBUTION ----------------
plt.figure(figsize=(5,4))
sns.countplot(x='status', data=data)
plt.title("Startup Status Distribution")
plt.show()

# ---------------- FUNDING VS STATUS ----------------
plt.figure(figsize=(6,4))
sns.boxplot(x='status', y='funding_total_usd', data=data)
plt.title("Funding vs Status")
plt.show()

# ---------------- VC IMPACT ----------------
plt.figure(figsize=(5,4))
sns.countplot(x='has_VC', hue='status', data=data)
plt.title("VC vs Startup Success")
plt.show()

# ---------------- CORRELATION ----------------
plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()