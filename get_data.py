from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
def get_data(filename):
    file = pd.read_csv(filename)
    row=1
    X=file.iloc[0, 1:].values
    for i in enumerate(file.itertuples()):    
        u = file.iloc[row, 1:].values
        row+=1
        X=np.vstack([X, u])
        if row==15:
            break   
    return(X)
X_1=get_data('nep.csv')
X_2=get_data('Vodka.csv')
X_3=get_data('HH.csv')
X=np.vstack([X_1, X_2, X_3])
print(X)

