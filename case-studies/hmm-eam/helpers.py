import os, sys
# for accessing src, stan, etc.
sys.path.append(os.path.abspath(os.path.join("../..")))
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pandas import read_csv
from src.models.HmmEam import model


def get_data(subject):
    path = os.path.join('dutilh-resources', 'data', subject) + '.csv'
    df = read_csv(path)
    df = np.array(df)

    return df[:400, 1], np.abs(df[:400, 2]-2.0)

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

def plot_joint_parameters(samples_one, samples_two, param_names, names = ["BayesFlow", "Stan"]):
    figsize = (1.5*len(param_names), 1.5*len(param_names))
    fig, axs = plt.subplots(nrows=len(param_names), ncols=len(param_names), figsize=figsize)

    bins = [
        np.linspace(np.min(samples_one[:,i]), np.max(samples_one[:,i]), 31) for i in range(len(param_names))
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
                axs[yi,xi].hist(samples_one[:,xi], bins=bins[xi],alpha = 0.5, density=True, color="skyblue")
                axs[yi,xi].hist(samples_two[:,xi], bins=bins[xi],alpha = 0.5, density=True, color="darkorange")
            elif xi > yi:
                axs[yi,xi].scatter(samples_one[:,xi], samples_one[:,yi], s=0.5, alpha=0.1, label="skyblue")
                axs[yi,xi].scatter(samples_two[:,xi], samples_two[:,yi], s=0.5, alpha=0.1, label="darkorange")
            else:
                axs[yi,xi].scatter(samples_one[:,xi], samples_one[:,yi], s=0.5, alpha=0.1, label="skyblue", zorder=2)
                axs[yi,xi].scatter(samples_two[:,xi], samples_two[:,yi], s=0.5, alpha=0.1, label="darkorange",zorder=1)

    axs[0,-1].legend(handles, names)
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

def plot_data(rts, choices):
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    axs.flatten()
    axs[0].set_title('Response times')
    axs[1].set_title('Responses')

    x = range(len(rts))
    axs[0].plot(x, rts)
    axs[1].scatter(x, choices, label = 'Data')
    running_mean = np.convolve(choices, np.ones(10)/10, mode='valid')
    axs[1].plot(x[:-9], running_mean, label = 'Running average')
    axs[1].legend()

    fig.tight_layout()

    return fig, axs


def plot_predictive_checks_histograms(classification, parameters, rts, choices):
    rng = np.random.default_rng()
    n_samples = parameters.shape[0]
    # replicate observations so that we can account for classification uncertainty
    states = rng.multinomial(1, classification).argmax(axis=-1)

    rts_rep = np.expand_dims(rts, 0)
    rts_rep = np.tile(rts_rep, (n_samples, 1))

    choices_rep = np.expand_dims(choices, 0)
    choices_rep = np.tile(choices_rep, (n_samples, 1))

    # draw predictives
    posterior_predictives = model.simulator(parameters)['sim_data']
    rts_pred = posterior_predictives[:,:,0]
    choices_pred = posterior_predictives[:,:,1]
    states_pred = posterior_predictives[:,:,2]

    # plot predictives vs data
    fig, axs = plt.subplots(2, 2)

    # plot response times
    bins = np.linspace(0, np.max(rts_pred), 46)
    axs[0,0].set_title("Guessing state")
    axs[0,0].hist(rts_pred[np.where(states_pred==0)], density=True, bins=bins, label="Posterior predictives", alpha=0.4)
    axs[0,0].hist(rts_rep[np.where(states==0)],       density=True, bins=bins, label="Observations", alpha=0.4)

    axs[1,0].set_title("Controlled state")
    axs[1,0].hist(rts_pred[np.where(states_pred==1)], density=True, bins=bins, label="Posterior predictives", alpha=0.4)
    axs[1,0].hist(rts_rep[np.where(states==1)],       density=True, bins=bins, label="Observations", alpha=0.4)

    axs[0,0].legend()
    axs[1,0].set_xlabel("Response times")

    # plot responses
    axs[0,1].set_title("Guessing state")
    axs[0,1].set_xlim([0, 1])
    axs[0, 1].hist(np.ma.array(choices_pred, mask=~(states_pred==0)).mean(axis=-1), density=True, label="Posterior predictives", alpha=0.5)
    axs[0, 1].hist(np.ma.array(choices_rep, mask=~(states==0)).mean(axis=-1), density=True, label="Observations", alpha=0.5)
    axs[0, 1].axvline(np.mean(choices_rep[np.where(states==0)]), label="Observations")

    axs[1,1].set_title("Controlled state")
    axs[1,1].set_xlim([0, 1])
    axs[1, 1].hist(np.ma.array(choices_pred, mask=~(states_pred==1)).mean(axis=-1), density=True, label="Posterior predictives", alpha=0.5)
    axs[1, 1].hist(np.ma.array(choices_rep, mask=~(states==1)).mean(axis=-1), density=True, label="Observations", alpha=0.5)
    axs[1, 1].axvline(np.mean(choices_rep[np.where(states==1)]), label="Observations")

    axs[0,1].legend()
    axs[1,1].set_xlabel("Average accuracy")


    fig.tight_layout()

    return fig, axs