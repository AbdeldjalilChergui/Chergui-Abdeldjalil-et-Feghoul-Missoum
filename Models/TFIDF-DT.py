import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
import pickle


df = pd.read_csv('XSS_dataset.csv', chunksize=1 )
df_list = []
for df in df :
    df_list.append(df)
df = pd.concat(df_list , sort = False )
if len(df) > 0 :
    print(f'Length of DataFrame {len(df)}, number of columns {len(df.columns)}, dimensions {df.shape}, number of elements {df.size}')
else :
    print('Problem loading DataFrame, DataFrame is empty.')
    
with open("tf_model.pkl", 'rb') as handle:
    tf1 = pickle.load(handle)

corpus = df['Sentence']

data = tf1.transform(corpus)
dense = data.todense()
denselist = dense.tolist()
df1 = pd.DataFrame(denselist)

df2 = df1
labels = df['Label']
df2["Label"]=labels

X = df2.drop('Label', axis=1).copy()
y = df2['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 ,random_state=42)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

plot_confusion_matrix(clf,
                      X_test, 
                      y_test, 
                      values_format='d', 
                      display_labels=["normal" , "attack"])

M = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = M.ravel() 

cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

precision=TP/(FP+TP)
print("Precision Score = ",precision)

recall=TP/(FN+TP)
print("Recall Score =  ",recall)

accuracy=(TP+TN)/(TP+FN+TN+FP)
print("Accuracy Score = ",accuracy)

fprate=FP/(TN+FP)
print("FP rate =", fprate)

fnrate=FN/(TP+FN)
print("FN rate =", fnrate)

f1=2* (precision*recall) / (precision+recall)
print("F1 Score = ",f1)

misclassification=(FP+FN)/(TP+TN+FP+FN)
print("Misclassification score =", misclassification)

joblib.dump(clf, "dt.pkl")










