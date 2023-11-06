import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load digits dataset
digits = load_digits()

# Split 30% into a seperate test set
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

# Train the (linear) SVM
clf = svm.SVC(kernel="linear", C=1)
clf.fit(x_train, y_train)

predictions = clf.predict(x_train)
correct = np.sum(predictions == y_train)
accuracy = correct / y_train.shape[0]
print("Train Set Accuracy: {:.2f}%".format(accuracy*100))

predictions = clf.predict(x_test)
correct = np.sum(predictions == y_test)
accuracy = correct / y_test.shape[0]
print("Test Set Accuracy: {:.2f}%".format(accuracy*100))

from matplotlib import pyplot as plt
x_wrong = x_test[predictions != y_test]
y_wrong = y_test[predictions != y_test]
pred_wrong = predictions[predictions != y_test]
n = x_wrong.shape[0]

rows = int(np.ceil(n))
fig, axs = plt.subplots(1,n)
for row in range(rows):
    idx = row
    if idx >= n:
        break
    axs[row].imshow(x_wrong[idx].reshape(8,8))

    target_class = y_wrong[idx]
    predicted_class = pred_wrong[idx]
    axs[row].set_title("GT: {} Pred: {}".
    format(target_class, predicted_class))

plt.show()