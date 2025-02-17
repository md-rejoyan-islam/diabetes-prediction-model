import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dt = pd.read_csv("./diabetes.csv")


print(dt.head(2))
print(dt.info())
print(dt.describe())

# check for null values in the dataset and print the sum of null values in each column
print(dt.isnull().sum())  # No null values

print("Shape of data:", dt.shape)  # (768, 9)

# columns in the dataset

# ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
columns = dt.columns

print("Columns:", columns)

# data types of columns

# All columns are of int64/floar64 type
print("Data types of columns:", dt.dtypes)

# draw histogram for Age column
dt['Age'].hist(bins=30, color='blue', alpha=0.7)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Age Vs Frequency Distribution', fontsize=16)
plt.show()

# draw age vs glucose level scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(dt['Age'], dt['Glucose'], c=range(
    len(dt['Age'])), cmap='viridis', alpha=0.7)
plt.colorbar().set_label('Index', fontsize=12)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Glucose', fontsize=12)
plt.legend(['Index'])
plt.title('Age vs Glucose Level', fontsize=16)
plt.show()

# histogram for all columns in the dataset
dt.hist(figsize=(10, 10))
plt.show()


# correlation matrix heatmap
corr_matrix = dt.corr(method='pearson')
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f",
            square=True, linewidths=1, annot_kws={"size": 5})
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.title('Correlation Matrix (Pearson method)', fontsize=16)  # Fixed title
plt.show()


# has any value zero in the columns
print((dt[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] == 0).sum())

# Replace zeros with median values in relevant columns
for col in columns:
    dt[col] = dt[col].replace(0, dt[col].median())


# modeling

X = dt.drop(['Outcome'], axis=1)
y = dt['Outcome']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Logistic Regression
log_reg = LogisticRegression(max_iter=500, random_state=42)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)


cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
            "No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
