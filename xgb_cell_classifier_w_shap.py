#!/usr/bin/env python
# coding: utf-8

# In[123]:


import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

path = "/Users/gustavosganzerla/Documents/covid/single_cell/rpc/filtered/with_celltype/"

list_y = []
list_x = []
for item in os.listdir(path):
  if item.endswith(".csv"):
    df = pd.read_csv(path+item)
    y_temp = df.iloc[:, 399].values
    x_temp = df.iloc[:, 2:398].values
    list_x.append(x_temp)
    list_y.append(y_temp)

y = np.concatenate(list_y)
x = np.concatenate(list_x)

df = pd.DataFrame(x,y)


# In[16]:


colnames = pd.read_csv("/Users/gustavosganzerla/Documents/covid/single_cell/rpc/filtered/with_celltype/patient01.csv")
columns_to_exclude = [0, 1, 398, 399]
colnames = colnames.drop(columns=colnames.columns[columns_to_exclude])
df['celltype'] = df.index
df = df.reset_index(drop = True)


# In[103]:


##Ran a Boruta to select the most distinctive features

X = df.drop(columns=['celltype'])
y = df['celltype']
y = y.values.ravel()
rf_classifier = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta_selector = BorutaPy(estimator=rf_classifier, n_estimators='auto', verbose=2, random_state=42)
boruta_selector.fit(X.values, y)
selected_feature_indices = np.where(boruta_selector.support_)[0]
# Get the corresponding column names of the selected features
selected_feature_names = X.columns[selected_feature_indices]


# In[ ]:


selected_indexes = np.array([31, 43, 64, 84, 124, 136, 141, 147, 159, 163, 183, 209, 276, 324, 330,
       331])
matching_names = all_features_names[selected_indexes]


# In[117]:


df_selected = df[selected_feature_names]
df_selected.columns = matching_names
df_selected['cell_type'] = y


# In[136]:


# Separate the feature matrix 'X' and the target variable 'y'
X = df_selected.iloc[:, 0:16]  # Columns 0 to 16 (0:17) as features
y = df_selected.iloc[:, -1]   # The last column (column 398) as the target variable 'y'



# In[ ]:


###ran a GridSearchCV to identify the best hyperparameters

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_classifier = xgb.XGBClassifier()
param_grid = {
    'n_estimators': [50, 100, 200], 
    'learning_rate': [0.01, 0.1, 0.2], 
    'max_depth': [3, 5, 7],             
}
grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

##the best hyperparameters are
#Best Hyperparameters: {'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 50}


# In[6]:


import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("/Users/gustavosganzerla/Documents/covid/single_cell/v2/boruta_data_celltype.csv")

X = df.iloc[:, 1:16]  # Columns 0 to 16 (0:17) as features
y = df.iloc[:, -1]   # The last column (column 398) as the target variable 'y'
y = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the XGBoost classifier
xgb_classifier = xgb.XGBClassifier(n_estimators=50, random_state=42, max_depth=5, learning_rate=0.2)

# OneVsRestClassifier with XGBoost classifier
ovr_classifier = OneVsRestClassifier(xgb_classifier)

# Fit the classifier to the training data
ovr_classifier.fit(X_train, y_train)

# Predict probabilities for each class
y_probs = ovr_classifier.predict_proba(X_test)

# Convert 'y' to categorical data type
y_test = pd.Categorical(y_test)
class_names = y_test.categories

# Compute ROC curve and ROC AUC score for each class
n_classes = len(class_names)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i, class_name in enumerate(class_names):
    y_test_binary = label_binarize(y_test, classes=class_names)
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class with class names in the legend
plt.figure(figsize=(10, 6))
for i, class_name in enumerate(class_names):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_name} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - One vs. All')
plt.legend(loc="lower right")
plt.show()


# In[189]:


import shap

for i in range(0,10):
    xgb_model = ovr_classifier.estimators_[i]  # Assuming the first estimator is the XGBoost model
    # Create an explainer object for the XGBoost model
    explainer = shap.Explainer(xgb_model)

    # Calculate SHAP values for all the instances in the test set
    shap_values = explainer.shap_values(X_train)

    # Plot the summary plot for all classes
    shap.summary_plot(shap_values, X_train, feature_names=X_train.columns)

