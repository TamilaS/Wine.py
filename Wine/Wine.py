import pandas as pd
df = pd.read_csv ("sparklingwine.csv", sep=";")


df.head(10)
df.describe()


def goodwine (quality):
    if quality >=6:
        return 1
    return 0


df ["good_wine"] = df.quality.apply(goodwine)
#df ["good_wine"] = df.quality.qpply (lambda x : 1 if x>=6 else 0)


import numpy as np
X = np.array (df[df.columns[:11]])
y=np.array(df.good_wine)


X_train_unproc = X[:400]
X_val_unproc = X[400:600]
X_test_unproc = X[600:]

y_train = y[:400]
y_val = y[400:600]
y_test = y[600:]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train_unproc)

X_train = scaler.transform (X_train_unproc)
X_val = scaler.transform (X_val_unproc)
X_test = scaler.transform (X_test_unproc)

from sklearn.neighbors import KNeighborsClassifier
clf= KNeighborsClassifier(3)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_val,y_val)

ks = range(1, 100, 1)
inSampleScores = []
valScores = []

for k in ks:
    clf = KNeighborsClassifier(k).fit(X_train,y_train)
    inSampleScores.append(clf.score(X_train,y_train))
    valScores.append(clf.score(X_val,y_val))

import matplotlib.pyplot as plt

p1 = plt.plot(ks, inSampleScores)
p2 = plt.plot(ks, valScores)
plt.legend(['in', 'out'], loc = 'upper right')

clf = KNeighborsClassifier(15).fit(X_train,y_train)
y_test_pred = clf.predict(X_test)
score_test = clf.score(X_test,y_test)
print(score_test)