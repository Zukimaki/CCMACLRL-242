import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


df = pd.read_csv("personality_dataset.csv")


print(df.head())


df.info()


print(df["Personality"].value_counts())


plt.hist(df["Followers"])
plt.xlabel("Followers")
plt.ylabel("Number of Samples")
plt.title("Distribution of Followers")
plt.show()


df = df.drop("Name", axis=1)




X = df.drop("Personality", axis=1).values


y = df["Personality"].values


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)



model = KNeighborsClassifier(n_neighbors=3)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)




cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)


precision = precision_score(y_test, y_pred, zero_division=0)
print("Precision:", precision)


recall = recall_score(y_test, y_pred, zero_division=0)
print("Recall   :", recall)


f1 = f1_score(y_test, y_pred, zero_division=0)
print("F1-score :", f1)



my_data = [[400, 20, 19]]

prediction = model.predict(my_data)

if prediction[0] == 0:
    print("Predicted Personality: Introvert")
else:
    print("Predicted Personality: Extrovert")