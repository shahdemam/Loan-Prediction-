import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
import math
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


data_set = pd.read_csv(r"C:\Users\user\Documents\Dataset_project.csv")
print("Dataset Preview:")
print(data_set.head())

  
independent = data_set.iloc[:, 1:-1].values
dependent = data_set.iloc[:, -1].values
print("Example of Independent Features:", independent[:5])
print("Example of Dependent Target:", dependent[:5])

#missing
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(independent[:, 5:9])
independent[:, 5:9] = imputer.transform(independent[:, 5:9])

 
independent = pd.DataFrame(independent)
columns = independent.columns
for column in columns:
    if independent[column].dtype == 'object':
        independent[column] = independent[column].infer_objects(copy=False)
        independent[column] = independent[column].fillna(independent[column].mode()[0])

print("Missing Values Check (After Processing):")
print(independent.isna().any(axis=0))

 
ct =ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(sparse_output=False), [0,1,2,3,4,9,10])],
    remainder='passthrough'
)
independent = np.array(ct.fit_transform(independent))
print("Encoded Features Preview:")
print(independent[:5])

le = LabelEncoder()
dependent = le.fit_transform(dependent)
print("Encoded Target Preview:")
print(dependent[:5])

 
independent_train, independent_test, dependent_train, dependent_test = train_test_split(
    independent, dependent, test_size=0.2, random_state=1
)
print("Training Data Size:", independent_train.shape)
print("Test Data Size:", independent_test.shape)

#scaling 
sc = StandardScaler()
independent_train[:, 5:9] = sc.fit_transform(independent_train[:, 5:9])
independent_test[:, 5:9] = sc.transform(independent_test[:, 5:9])

#impalanced
target_count=data_set.Loan_Status.value_counts()
print('class0:',target_count.iloc[0])
print('class1:',target_count.iloc[1])
print('Propotion:',round(target_count.iloc[0]/target_count.iloc[1],2), ':1')
target_count.plot(kind='bar',title='count(target)')
plt.show()


 
count_class_yes, count_class_no =data_set['Loan_Status'].value_counts()
df_class_yes =data_set [data_set ['Loan_Status'] == 'Y']
df_class_no =data_set [data_set ['Loan_Status'] == 'N']
df_class_no_upsampled =resample(df_class_no,replace=True,n_samples=count_class_yes,random_state=42)
df_upsampled = pd.concat([df_class_no_upsampled, df_class_yes])
df_upsampled['Loan_Status'].value_counts().plot(kind='bar', title='Count (Target)');
plt.show()

#model naitv bayes
model = GaussianNB()
model.fit(independent_train, dependent_train)
dependent_pred = model.predict(independent_test)

 
# cm = confusion_matrix(dependent_test, dependent_pred)
# print("Confusion Matrix:\n", cm)

 
# print("Accuracy:", accuracy_score(dependent_test, dependent_pred))
# print("\nClassification Report:")
# print(classification_report(dependent_test, dependent_pred))
 
#model knn
key=math.ceil((math.sqrt(len(dependent_test)))-1)
model=KNeighborsClassifier(n_neighbors=key)
model.fit(independent_train, dependent_train)
dependent_pred= model.predict(independent_test) 
print("Accuracy:", accuracy_score(dependent_test,dependent_pred))
print("Confusion Matrix:\n", confusion_matrix(dependent_test,dependent_pred))
print("Classification Report:\n", classification_report(dependent_test, dependent_pred))
cm=confusion_matrix(dependent_test, dependent_pred)
# disp=ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()

 
ax = sns.heatmap(cm, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values')
ax.xaxis.set_ticklabels(['Not Loan', 'Loan'])
ax.yaxis.set_ticklabels(['Not Loan', 'Loan'])
plt.show()

# actual and predict
results = pd.DataFrame({
    "Actual Values": dependent_test,
    "Predicted Values": dependent_pred
})
results["Correct"] = results["Actual Values"] == results["Predicted Values"]
print("\nResults Table:")
print(results.head(15))

#inut to predict
sample_input = input("Enter a sample (comma-separated values matching the independent feature format): ")
sample_input = np.array(sample_input.split(","), dtype=object).reshape(1, -1)

 
sample_input_encoded = np.array(ct.transform(sample_input))
sample_input_encoded[:, 5:9] = sc.transform(sample_input_encoded[:, 5:9])

 
sample_prediction = model.predict(sample_input_encoded)
predicted_label = le.inverse_transform(sample_prediction)
print("Prediction for the input sample:", predicted_label[0])
