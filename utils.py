import numpy as np
import matplotlib.pyplot as plt

def normalize_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def plot_spectrogram(spectrogram, title=None):
    fig, ax = plt.subplots()
    im = ax.imshow(spectrogram, aspect='auto', origin='lower', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    return fig

def save_plot(fig, filename):
    fig.savefig(filename)
    plt.close(fig)