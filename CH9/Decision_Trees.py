import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def main():
    data=pd.read_csv("./CH9/Admission_Predict.csv")
    features=data.drop(['Serial No.','Chance of Admit'],axis=1)
    labels=data['Chance of Admit']
    dt_regressor=DecisionTreeRegressor(max_depth=3,min_samples_leaf=10,min_samples_split=10)
    dt_regressor.fit(features,labels)
    result=dt_regressor.predict([[320,110,3,4.0,3.5,8.9,0]])
    print(f"The admit ratio is {result}")

if __name__ == "__main__":
    main()