import math
import numpy as np
from matplotlib import pyplot as plt

# In dieser Aufgabe wollen wir mit verschiedenen Loss-Funktionen experimentieren.
# Dazu betrachten wir ein lineares Model der Form
#
#   y = w0 * x +  w1

# Aufgabe 1
# Um später das Modell anwenden zu könne müssen wir Prädiktionen machen können
# Sei dazu X ein numpy.array mit den Datenpunkten. Wenden Sie das (ebenfalls übergebene)
# Model auf die Daten an und geben Sie die jeweilige Prädiktion zurück
def predict(model, x):
    return model[0] *x + model[1]

# Aufgabe 2
# Implementieren Sie die "evaluate"-Methode der unteren Klasse "MSELoss", wo sie den 
# MSE Loss für ihre prädizierten Residuuen berchnen sollen. 
class MSELoss:
    def __init__(self):
        pass

    def evaluate(self, model, x, y):
        # Hinweis: Berechnen Sie zunächst das Residuum
        residuum = predict(model, x) - y

        # Berechnen Sie dann den (summarischen, also skalaren) Fehlerterm
        error = np.sum(residuum**2)

        # Und dann den Gradienten
        grad = np.array([
            np.sum(residuum * x),
            np.sum(residuum * 1)
        ])

        # Geben Sie sowohl den (skalaren) Fehler als den (vektoriellen) Gradienten zurück
        return error, grad
    
# Aufgabe 3
# Implementieren Sie nun eine ähnliche Klasse "MAE-Loss", sowie die "LogCosh-Loss" Klasse
# Berechnen Sie dort die jeweiligen Loss-Terme und Gradienten.
# Tauschen Sie dann unten im Code den MSELoss durch den MAELoss bzw. LogCoshLoss aus und beobachten
# sie was passiert
class MAELoss:
    def __init__(self):
        pass

    def evaluate(self, model, x, y):
        residuum = predict(model, x) - y

        error = np.sum(np.abs(residuum))

        grad = np.array([
            np.sum(np.sign(residuum) * x),
            np.sum(np.sign(residuum) * 1)
        ])

        return error, grad    

class LogCoshLoss:
    def __init__(self):
        pass

    def evaluate(self, model, x, y):
        residuum = predict(model, x) - y

        error = np.sum(np.log(np.cosh(residuum)))

        grad = np.array([
            np.sum(np.tanh(residuum) * x),
            np.sum(np.tanh(residuum) * 1)
        ])

        return error, grad     

# Bei der folgenden Funktion erhalten Sie als Parameter ein Axis-Object von MatPlotLib sowie einer ihrer
# obigen Loss-Klassen, die unabhänigen Daten (x) und die abhänige Variable (y). 
#
# Wir wollen die Fehlerfunktion in Abhängigkeit der Modellparameter zeichen.
# Dazu wählen wir mit np.linspace(-10, 10, 25) verschiedene mögliche Werte für w0 und w1,
# berechnen dort jeweils den Loss und zeichnen mit axs.contourf einen Konturplot
def plot_error_surface(axs, loss, x, y):
    w0_range = np.linspace(-10, 10, 25)
    w1_range = np.linspace(-10, 10, 25)
    
    xx, yy = np.meshgrid(w0_range, w1_range)
    z = np.zeros_like(xx)
    for idx0, w0 in enumerate(w0_range):
        for idx1, w1 in enumerate(w1_range):
            error, _ = loss.evaluate(np.array([w0, w1]), x, y)
            z[idx1, idx0] = error

    axs.contourf(xx, yy, z, levels=250, cmap="jet")

if __name__ == "__main__":
    model = np.array([9.0, 9.0])
    x = np.array([ 0.5, 1.0, 2.0, 3.0, 2.5])
    y = np.array([ 1.0, 5.0, 4.0, 3.0, 1.0])

    print(f"Model: y = {model[0]:.1f}x+{model[1]:.1f}")
    print("Prädiktion: ", predict(model, x))

    eta = 0.05
    steps = 200

    loss = MSELoss()

    loss_values = []

    plt.ion()

    fig, axs = plt.subplots(1, 3)
    plot_error_surface(axs[2], loss, x, y)

    while True:    
        x_ = np.linspace(0, 5, 20)
        y_ = predict(model, x_)

        axs[0].cla()
        axs[0].plot(loss_values)
        axs[0].set_xlim(0, steps)
        axs[0].set_ylim(0, 100)
        axs[0].set_title("Loss")

        axs[1].cla()
        axs[1].plot(x, y, 'g.')
        axs[1].plot(x_, y_, 'k')
        axs[1].set_xlim(0, 5)
        axs[1].set_ylim(0, 10)
        axs[1].set_title("Model")

        axs[2].plot(model[0], model[1], 'wD')
        
        error, grad = loss.evaluate(model, x, y)
        loss_values.append(error)

        model = model - eta * grad

        if np.sum(grad**2) < 0.01:
            break

        fig.show()
        plt.pause(0.01)

    plt.ioff()
    plt.show()
