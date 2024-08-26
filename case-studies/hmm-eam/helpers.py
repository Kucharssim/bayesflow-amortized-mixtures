import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_marginal_posteriors(stan, bayesflow, param_names):

    fig, axs = plt.subplots(ncols=4, nrows=2)
    axs = axs.flatten()

    bins = [
        np.linspace(np.min(bayesflow[:,i]), np.max(bayesflow[:,i]), 31) for i in range(len(param_names))
    ]
    for i, par in enumerate(param_names):
        axs[i].hist(bayesflow[:,i], bins = bins[i], alpha=0.5, density=True, label = "BayesFlow")
        axs[i].hist(stan[:,i], bins = bins[i], alpha=0.5, density=True, label = "Stan")
        axs[i].set_title(par)

    axs[0].legend()
    fig.tight_layout()

    return fig, axs

def plot_joint_posteriors(stan, bayesflow, param_names):
    figsize = (1.5*len(param_names), 1.5*len(param_names))
    fig, axs = plt.subplots(nrows=len(param_names), ncols=len(param_names), figsize=figsize)

    bins = [
        np.linspace(np.min(bayesflow[:,i]), np.max(bayesflow[:,i]), 31) for i in range(len(param_names))
    ]

    handles = [
        patches.Patch(color='skyblue'),
        patches.Patch(color='darkorange')
    ]
    for xi, x_par in enumerate(param_names):
        axs[-1,xi].set_xlabel(x_par)
        for yi, y_par in enumerate(param_names):
            if xi == yi:
                axs[yi,0].set_ylabel(y_par)
                axs[yi,xi].hist(bayesflow[:,xi], bins=bins[xi],alpha = 0.5, density=True, color="skyblue")
                axs[yi,xi].hist(stan[:,xi], bins=bins[xi],alpha = 0.5, density=True, color="darkorange")
            elif xi > yi:
                axs[yi,xi].scatter(bayesflow[:,xi], bayesflow[:,yi], s=0.5, alpha=0.1, label="skyblue")
                axs[yi,xi].scatter(stan[:,xi], stan[:,yi], s=0.5, alpha=0.1, label="darkorange")
            else:
                axs[yi,xi].scatter(bayesflow[:,xi], bayesflow[:,yi], s=0.5, alpha=0.1, label="skyblue", zorder=2)
                axs[yi,xi].scatter(stan[:,xi], stan[:,yi], s=0.5, alpha=0.1, label="darkorange",zorder=1)

    axs[0,-1].legend(handles, ["BayesFlow", "Stan"])
    fig.tight_layout()

    return fig, axs

def plot_classification(classification, suptitle, methods = ['BayesFlow', 'Stan']):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs.flatten()
    axs[0].set_title('P(Guessing)')
    axs[1].set_title('P(Controlled)')

    x = range(classification.shape[-2])
    for i, method in enumerate(methods):
        for j, s in enumerate(['Guessing', 'Controlled']):
            axs[j].plot(x, classification[1,i,:,j], label = method)
            axs[j].fill_between(x, classification[0,i,:,j], classification[2,i,:,j])


    axs[1].legend(loc='upper left')
    fig.suptitle(suptitle)
    fig.tight_layout()

    return fig, axs
