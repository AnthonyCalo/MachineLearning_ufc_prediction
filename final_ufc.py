import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_gamma_deviance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats

ufc = pd.read_csv("./ufcData.csv")


ufc['date'] = pd.to_datetime(ufc["date"], format="%Y-%m-%d")
# ufc = ufc.loc[ufc["date"]> datetime.strptime("01/01/2010", "%m/%d/%Y")]

# ufc=ufc[["B_current_win_streak","R_current_win_streak","B_age", "R_age","Winner"]]

ufc["B_winper"] = ufc['B_wins'] / (ufc["B_wins"] + ufc["B_losses"])*100
ufc["B_winper"].fillna(0, inplace=True)
ufc["R_winper"] = ufc['R_wins'] / (ufc["R_wins"] + ufc["R_losses"])*100
ufc["R_winper"].fillna(0)
ufc["B_winper"].fillna(0, inplace=True)
ufc = ufc.drop(columns=["B_wins", "B_losses", "R_wins", "R_losses"])
winner_encoder = LabelEncoder()
stance_encoder = LabelEncoder()
ufc["Winner"] = winner_encoder.fit_transform(ufc["Winner"])
# ufc["R_Stance"] = winner_encoder.fit_transform(ufc["R_Stance"])
# ufc["B_Stance"] = winner_encoder.fit_transform(ufc["B_Stance"])

ufc=ufc[["B_current_win_streak","R_current_win_streak","B_age", "R_age","Winner"]]

print(len(ufc["Winner"]))
print(ufc.isnull().any())
ufc.dropna(inplace=True)
print(len(ufc["Winner"]))

x = ufc.iloc[:, [0, 1, 2,3]].values
y = ufc.iloc[:,4].values
# x=ufc.drop(columns=["Winner","location", "title_bout", "weight_class", "R_fighter", "B_fighter", "Referee","date" ]).values
# x=ufc.drop(columns=['Winner']).values
# y=ufc["Winner"].values

winnersAge=[]
losersAge=[]
#graph age and get ttest
for i,r in ufc.iterrows():
    if(r["Winner"]==1):
        continue
    elif(r["Winner"]==0):
        winnersAge.append(r["R_age"])
        losersAge.append(r["B_age"])
    else:
        winnersAge.append(r["B_age"])
        losersAge.append(r["R_age"])

# print("T test",stats.ttest_ind(a=winnersAge, b=losersAge, equal_var=False))


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)


sc = StandardScaler()

x_train =sc.fit_transform(x_train)

x_test = sc.transform(x_test)

gnb = GaussianNB()

gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)


cm = confusion_matrix(y_test, y_pred)
print(cm)
correct=0
for x,y in zip(y_pred, y_test):
    if(x==y):
        correct+=1
    
print("Naive Bayes Results: \n")
print("Accuracy: {}".format(correct/len(y_test)))
sensitivity = precision_score(y_test, y_pred,average='micro' )
specificity = recall_score(y_test, y_pred, average='micro')
print("Sensitivity: {}".format(sensitivity))
print("Specificity: {}".format(specificity))



lr_model = LogisticRegression()

lr_model.fit(x_train, y_train)

y_pred = lr_model.predict(x_test)
print("\nLogistic Regression Results\n")
# print("Accuracy score for Logistic regression? is :{}".format(accuracy_score(y_test, y_pred)))

sensitivity = precision_score(y_test, y_pred,average='micro' )
specificity = recall_score(y_test, y_pred, average='micro')

print("Sensitivity: {}".format(sensitivity))
print("Specificity: {}".format(specificity))



random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)

Y_prediction = random_forest.predict(x_test)

# random_forest.score(x_train, y_train)

print("\nRandom Forest Model results\n")
sensitivity = precision_score(y_test, Y_prediction,average='micro' )
specificity = recall_score(y_test, Y_prediction, average='micro')

print("Sensitivity: {}".format(sensitivity))
print("Specificity: {}".format(specificity))


