from matplotlib import pyplot as plt
import numpy as np

# In diesem Beispiel möchten wir je 2000 Datenpunkte
# aus einer Normalverteilung (mit unterschiedlichen Mittelwerten)
# anhand eines einfachen Schwellwerten klassifizieren.
# Das bedeutet, wir klassifizieren alle Datenpunkte als positive, die größer sind als der Schwellwert (threshold)
# und alle, die kleiner sind als negativ.
positives = np.random.normal( 1.0, 1.0, 5000)
negatives = np.random.normal(-1.0, 1.0, 20000)

threshold = 0.0
plt.hist(positives, np.linspace(-5.0, 5.0, 60), color='b', alpha=0.5)
plt.hist(negatives, np.linspace(-5.0, 5.0, 60), color='r', alpha=0.5)
plt.plot([threshold, threshold], [0, 160], 'k')

# Aufgabe 1
# Zählen sie die true positives, die true negatives, die false positives und die false negatives
true_positives = np.sum(positives > threshold)
true_negatives = np.sum(negatives < threshold)
false_positives = np.sum(negatives > threshold)
false_negatives = np.sum(positives < threshold)

# Aufgabe 2
# Berechnen Sie die TPR, TNR, Accuracy, Balanced Accuracy sowie die Precision (vgl. Skript)
n_positives = positives.shape[0]
n_negatives = negatives.shape[0]

tpr = true_positives / (true_positives + false_negatives)
tnr = true_negatives / (true_negatives + false_positives)
acc = (true_positives + true_negatives) / (n_positives + n_negatives)
bacc = (tpr + tnr) / 2
precision = true_positives / (true_positives + false_positives)

print(f"True Positive Rate (recall): {tpr*100.0:.2f}%")
print(f"True Negative Rate: {tnr*100.0:.2f}%")
print(f"Accuracy: {acc*100.0:.2f}%")
print(f"Balanced Accuracy: {bacc*100.0:.2f}%")
print(f"Precision: {precision*100.0:.2f}%")

plt.show()

# Aufgabe 3
# Iterieren Sie nun mit np.linspace(-5,5,100) über verschiedene mögliche Schwellwerte, berechnen Sie jeweils die TPR (Recall) sowie
# die Precision (vgl. oben) und speichern sie diese beiden Werte in je einem Python Array. Plotten Sie dann die TPR gegen die Precision
# in einer Kurve.
# Hinweis: Invertieren Sie die x-axis mit Hilfe von plt.gca().invert_axis() um den typischen Verlauf der s.g. ROC Kurve zu erhalten. 
tpr_array = []
pr_array = []
for threshold in np.linspace(-5, 5, 100):
    true_positives = np.sum(positives > threshold)
    true_negatives = np.sum(negatives < threshold)
    false_positives = np.sum(negatives > threshold)
    false_negatives = np.sum(positives < threshold)

    tpr = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    tpr_array.append(tpr)
    pr_array.append(precision)

plt.plot(tpr_array, pr_array, 'r')
plt.xlim(0,1)
plt.ylim(0,1)
plt.gca().invert_xaxis()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()


