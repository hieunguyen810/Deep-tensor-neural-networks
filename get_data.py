from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
def get_data(filename, label):
    file = pd.read_csv(filename)
    row=1
    X=file.iloc[0, 1:].values
    y=label
    t = X[0]/2
    for i in enumerate(file.itertuples()):    
        u = file.iloc[row, 1:].values 
        row+=1
        if u[0]>t:
             X=np.vstack([X, u])
             y=np.append(y,label)
        if row==10:
            break
    return X, y
X, y=get_data('File do/HH1.csv', 0)
#X=np.vstack([X_1, X_2, X_3])
y = np.append(y_1, y_2)
y = np.append(y, y_3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("X_train",X_train)
print("y_test:", y_test)

