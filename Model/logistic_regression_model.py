import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score,roc_curve, auc, f1_score
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the preprocessed dataset
df = pd.read_csv('/home/binjaminni@mta.ac.il/thinclient_drives/PreProcess/processed_data.csv')


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


best_model = None
best_recall = 0
best_f1 = 0
best_avg_score = 0
best_conf_mat = None
best_params = None
best_th = None




print(f"________Start the experiments:________ ")

for pen in penalties:
    for C in Cs:
        for max_iter in max_iters:
            logistic_model = LogisticRegression(penalty = pen, C=C, max_iter = max_iter, solver=solver, random_state=42)
            logistic_model.fit(X_train, y_train)

            # Calculate the probabilities of the positive class
            y_probs = logistic_model.predict_proba(X_test)[:,1]

            # Calculate the ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_probs)

            # Calculate the AUC
            roc_auc = auc(fpr, tpr)

            # Plot the ROC curve
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()

            # Find the optimal threshold
            optimal_idx = np.argmin(np.sqrt((1-tpr)**2 + fpr**2))
            optimal_threshold = thresholds[optimal_idx]
            print("Optimal threshold value:", optimal_threshold)

            y_pred = np.where(y_probs >= optimal_threshold,1,0)
            recall = recall_score(y_test,y_pred)
            accuracy = accuracy_score(y_test,y_pred)
            f1=f1_score(y_test,y_pred)
            avg_score = (recall + f1) / 2
            mat = confusion_matrix(y_test,y_pred)

            # Check if the current model's average score is better than the best one found so far
            if avg_score > best_avg_score:
                print("___________New best model found!________")
                best_model = logistic_model
                best_avg_score = avg_score
                best_recall = recall
                best_f1 = f1
                best_conf_mat = mat
                best_params = (pen, C, max_iter)
                best_th = optimal_threshold


            print(f"Parameters: penalty:{pen}, C:{C}, Max Iterations{max_iter}, threshold:{optimal_threshold}")
            print(f"recall: {recall}")
            print(f"Accuracy: {accuracy}")
            print(f"F1 Score: {f1}")
            print(mat)
            print()
        print("_______________________________________________________")

print("Best Model Parameters: ", best_params)
print(f"Best Model F1 Score: {best_f1}")
print(f"Best Model Recall Score: {best_recall}")
print("Best Model Average Score: ", best_avg_score)
print("Best Model Confusion Matrix: ", best_conf_mat)
print("Best Model Threshold: ", best_th)

# Save the best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save the best parameters and threshold
with open('best_params.txt', 'w') as f:
    f.write(f"Best Model Parameters: {best_params}\n")
    f.write(f"Best Model Threshold: {best_th}\n")
    f.write(f"Best Model F1 Score: {best_f1}\n")
    f.write(f"Best Model Recall Score: {best_recall}\n")