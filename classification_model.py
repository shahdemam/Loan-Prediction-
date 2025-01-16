import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Import dataset
data_set = pd.read_csv(r"C:\Users\user\Documents\Dataset_project.csv")
print(data_set.head())

# Independent // Dependent data (Feature and Target)
independent = data_set.iloc[:, 1:-1].values
dependent = data_set.iloc[:, -1].values
print('Independent:', independent[0:11])
print('Dependent: ', dependent[0:11])

# Check for missing data and data types
print(data_set.isna().any(axis=0))
print(data_set.dtypes)

# Handle missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(independent[:, 5:9])
independent[:, 5:9] = imputer.transform(independent[:, 5:9])

independent = pd.DataFrame(independent)
columns = independent.columns
for column in columns:
    if independent[column].dtype == 'object':
        independent[column] = independent[column].infer_objects()
        independent[column] = independent[column].fillna(independent[column].mode()[0])

print(independent.isna().any(axis=0))
independent = independent.values
print("Example of Handle: ", independent[11])

# Encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), [0, 1, 2, 3, 4, 9, 10])], remainder='passthrough')
independent = np.array(ct.fit_transform(independent))
data_encoded = pd.DataFrame(independent)
print(data_encoded.head(12))
print(data_encoded.iloc[0:20, 0:17])

le = LabelEncoder()
dependent = le.fit_transform(dependent)
target = pd.DataFrame(dependent)
print("target",target.head(21))
print("Ex of encoder: ", dependent[0:11])

# Split data into test and train sets
independent_train, independent_test, dependent_train, dependent_test = train_test_split(independent, dependent, test_size=0.2, random_state=1)
print('independent_train', independent_train[0:15])
print('independent_test', independent_test[0:15])
print('dependent_train', dependent_train[0:15])
print('dependent_test', dependent_test[0:15])

# Standardization
sc = StandardScaler()
independent_train[:, 5:9] = sc.fit_transform(independent_train[:, 5:9])
independent_test[:, 5:9] = sc.transform(independent_test[:, 5:9])
print("independent_train standardization", independent_train[0:10, 5:9])
print("independent_test standardization", independent_test[0:10, 5:9])

# Visualization of imbalance or balance
target_count = data_set.Loan_Status.value_counts()
print('class_Yes:', target_count.iloc[0])
print('class_No:', target_count.iloc[1])
print('Proportion:', round(target_count.iloc[0] / target_count.iloc[1], 2), ':1')
target_count.plot(kind='bar', title='Count(target)')
plt.show()

# Handle imbalance
count_class_yes, count_class_no = data_set['Loan_Status'].value_counts()
df_class_yes = data_set[data_set['Loan_Status'] == 'Y']
df_class_no = data_set[data_set['Loan_Status'] == 'N']
df_class_no_upsampled = resample(df_class_no, replace=True, n_samples=count_class_yes, random_state=42)
df_upsampled = pd.concat([df_class_no_upsampled, df_class_yes])
df_upsampled['Loan_Status'].value_counts().plot(kind='bar', title='Count (Target)')
plt.show()
  

# knn model
# key=math.ceil((math.sqrt(len(dependent_test)))-1)
# model=KNeighborsClassifier(n_neighbors=key)
# model.fit(independent_train, dependent_train)
# dependent_pred= model.predict(independent_test) 
# print("Accuracy:", accuracy_score(dependent_test,dependent_pred))
# print("Confusion Matrix:\n", confusion_matrix(dependent_test,dependent_pred))
# print("Classification Report:\n", classification_report(dependent_test, dependent_pred))
# cm=confusion_matrix(dependent_test, dependent_pred)

 
 


# Naive Bayes model
model = GaussianNB()
model.fit(independent_train, dependent_train)
dependent_pred = model.predict(independent_test)
print("Confusion Matrix:\n", confusion_matrix(dependent_test, dependent_pred))
print("Accuracy:", accuracy_score(dependent_test, dependent_pred))
cm = confusion_matrix(dependent_test, dependent_pred)
print("\nClassification Report:")
print(classification_report(dependent_test, dependent_pred))

# Visualization of confusion matrix
ax = sns.heatmap(cm, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix' + '\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['Not Loan', 'Loan'])
ax.yaxis.set_ticklabels(['Not Loan', 'Loan'])
plt.show()

# Results Table
results = pd.DataFrame({
    "Actual Values": dependent_test,
    "Predicted Values": dependent_pred
})
results["Correct"] = results["Actual Values"] == results["Predicted Values"]
print("\nResults Table:")
print(results)

# Allow user input for prediction
sample_input = input("Enter a sample (comma-separated values matching the independent feature format): ")
sample_input = np.array(sample_input.split(","), dtype=object).reshape(1, -1)

# Process the sample (encoding and standardizing)
sample_input_encoded = np.array(ct.transform(sample_input))
sample_input_encoded[:, 5:9] = sc.transform(sample_input_encoded[:, 5:9])

# Predict using the model
sample_prediction = model.predict(sample_input_encoded)
predicted_label = le.inverse_transform(sample_prediction)
print("Prediction for the input sample:", predicted_label[0])    

 
