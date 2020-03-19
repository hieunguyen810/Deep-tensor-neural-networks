from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
file_1 = pd.read_csv('nep.csv')
file_2 = pd.read_csv('Vodka.csv')
file_3 = pd.read_csv('HH.csv')
# nep
row_1=1
X_1=file_1.iloc[0, 1:].values
for i in enumerate(file_1.itertuples()):    
    u_1 = file_1.iloc[row_1, 1:].values
    row_1+=1
    X_1=np.vstack([X_1, u_1])
    if row_1==15:
        break   
print(X_1)
# volka
row_2=1
X_2=file_1.iloc[0, 1:].values
for j in enumerate(file_2.itertuples()):    
    u_2 = file_2.iloc[row_2, 1:].values
    row_2+=1
    X_2=np.vstack([X_2, u_2])
    if row_2==15:
        break   
print(X_2)
# HH
row_3=1
X_3=file_3.iloc[0, 1:].values
for k in enumerate(file_3.itertuples()):    
    u_3 = file_3.iloc[row_3, 1:].values
    row_3+=1
    X_3=np.vstack([X_3, u_3])
    if row_3==15:
        break   
print(X_3)
X=np.vstack([X_1, X_2, X_3])
print(X) 
neigh = KNeighborsClassifier(n_neighbors=4)
y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
neigh.fit(X, y)
test=file_3.iloc[3, 1:].values
print(np.array([test]))
print(neigh.predict(np.array([test])))
