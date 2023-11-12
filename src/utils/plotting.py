import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

def save_fig(ax,file_path):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt = ax.get_figure()
    plt.savefig(file_path, transparent=True)
    return

def gen_confusion_matrix(model, y_true, y_pred):
    fig = plt.figure()
    cm = confusion_matrix(y_true,y_pred) 
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xticks(np.arange(0,len(model.classes_)), [str(i) for i in model.classes_])
    plt.yticks(np.arange(0,len(model.classes_)), [str(i) for i in model.classes_])
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    return fig

def gen_parity_plot(model, y_true, y_pred):
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
                                                     (0, '#ffffff'),
                                                     (1e-20, '#440053'),
                                                     (0.2, '#404388'),
                                                     (0.4, '#2a788e'),
                                                     (0.6, '#21a784'),
                                                     (0.8, '#78d151'),
                                                     (1, '#fde624'),
                                                     ], N=256)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(y_true, y_pred, cmap=white_viridis)
    fig.colorbar(density, label='Number of points per pixel')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('True Yield')
    ax.set_ylabel('Predicted Yield')
    plt.plot([0, 1], [0, 1], linestyle='dashed', color='black') # plot the y=x line
    mean_guess = np.mean(y_true)
    print(mean_guess)
    plt.plot([0, 1], [mean_guess, mean_guess], color='red')
    return fig