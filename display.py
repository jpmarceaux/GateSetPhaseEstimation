import numpy as np
import matplotlib.pyplot as plt

def plot_complex_signal_in_unit_circle(signal, ax=None, title=None):
    """Plots a complex signal in the unit circle
    Use a viridis color gradient to show the evolution of the signal, 
    the signal is plotted as points inside a solid circule of radius 1
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot(np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)), color='black')
    ax.scatter(signal.real, signal.imag, c=np.linspace(0, 1, len(signal)), cmap='viridis')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Re')
    ax.set_ylabel('Im')
    if title is not None:
        ax.set_title(title)

def plot_cartan_signals(edesign, germ, params):
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    subspace = (0, 1)
    signal = edesign.signal(germ, params, subspace)
    plot_complex_signal_in_unit_circle(signal, ax=axs[0, 0], title='(0,1)')

    subspace = (0, 2)
    signal = edesign.signal(germ, params, subspace)
    plot_complex_signal_in_unit_circle(signal, ax=axs[0, 1], title='(0,2)')

    subspace = (0, 3)
    signal = edesign.signal(germ, params, subspace)
    plot_complex_signal_in_unit_circle(signal, ax=axs[0, 2], title='(0,3)')

    subspace = (1, 2)
    signal = edesign.signal(germ, params, subspace)
    plot_complex_signal_in_unit_circle(signal, ax=axs[1, 0], title='(1,2)')

    subspace = (1, 3)
    signal = edesign.signal(germ, params, subspace)
    plot_complex_signal_in_unit_circle(signal, ax=axs[1, 1], title='(1,3)')

    subspace = (2, 3)
    signal = edesign.signal(germ, params, subspace)
    plot_complex_signal_in_unit_circle(signal, ax=axs[1, 2], title='(2,3)')

    # super title from germ
    plt.suptitle(f'{germ}')

     # tight layout
    plt.tight_layout()