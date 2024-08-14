import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score 
import numpy as np

# Load the preprocessed dataset
df = pd.read_csv('processed_data.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['eid', 'tag'])
y = df['tag']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train))


penalties = ['l1', 'l2']
Cs = [0.1, 1, 10]
max_iters = [500,1000]
solver = 'liblinear'
ths = [0.1, 0.3, 0.5, 0.7]

best_model = None
best_recall = 0 
best_conf_mat = None
best_params = None
best_th = None




for pen in penalties:
    for C in Cs:
        for max_iter in max_iters:
            logistic_model = LogisticRegression(penalty = pen, C=C, max_iter = max_iter, solver=solver, random_state=42)
            logistic_model.fit(X_train, y_train)
            for t in ths:
                y_probs = logistic_model.predict_proba(X_test)[:,1]
                y_pred = np.where(y_probs >= t,1,0)
                recall = recall_score(y_test,y_pred)
                accuracy = accuracy_score(y_test,y_pred)
                mat = confusion_matrix(y_test,y_pred)
                print(f"Parameters: penalty:{pen}, C:{C}, Max Iterations{max_iter}, threshold:{t}")
                print(f"recall: {recall}")
                print(f"Accuracy: {accuracy}")
                print(mat)
                print()
            print("_______________________________________________________")
    