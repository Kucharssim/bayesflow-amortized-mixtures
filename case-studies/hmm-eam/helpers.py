import os, sys
# for accessing src, stan, etc.
sys.path.append(os.path.abspath(os.path.join("../..")))
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import numpy as np
import pickle
from pandas import read_csv
from src.models.HmmEam import model


def get_data(subject):
    path = os.path.join('dutilh-resources', 'data', subject) + '.csv'
    df = read_csv(path)
    df = np.array(df)

    return df[:400, 1], np.abs(df[:400, 2]-2.0)

def get_bayesflow_samples(subject):
    path = os.path.join('dutilh-resources', 'bayesflow-fits', subject) + '.pkl'
    with open(path, 'rb') as f:
        posterior, (forward, backward, smoothing ) = pickle.load(f)
        posterior = posterior[0]
        forward = forward[0]
        backward = backward[0]
        smoothing = smoothing[0]
    
    return (posterior, (forward, backward, smoothing))

def get_stan_samples(subject):
    path = os.path.join('dutilh-resources', 'stan-fits', subject) + '.pkl'
    with open(path, 'rb') as f:
        stan_fit = pickle.load(f)

    stan_posterior = np.array([
        stan_fit.stan_variable('transition_matrix')[:, 0, 0],
        stan_fit.stan_variable('transition_matrix')[:, 1, 1],
        stan_fit.stan_variable('alpha_1'),
        stan_fit.stan_variable('alpha_2'),
        stan_fit.stan_variable('nu_1'),
        stan_fit.stan_variable('nu_2')[:,0],
        stan_fit.stan_variable('nu_2')[:,1],
        stan_fit.stan_variable('tau'),
        ]).transpose()
    
    return stan_posterior, stan_fit

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

def plot_marginal_samples(param_names, colors=None, **samples):
    fig, axs = plt.subplots(ncols=4, nrows=2)
    axs = axs.flatten()

    min = np.min([np.min(s, axis=0) for s in samples.values()], axis=0)
    max = np.max([np.max(s, axis=0) for s in samples.values()], axis=0)

    bins = [np.linspace(min[i], max[i], 31) for i in range(len(param_names))]

    for i, par in enumerate(param_names):
        axs[i].xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        axs[i].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

        for k, s in samples.items():
            axs[i].hist(s[:,i], bins=bins[i], alpha=0.5, density=True, label=k, color=colors[k])
        axs[i].set_title(par)

    fig.subplots_adjust(right=0.8)
    handles = [patches.Patch(color=colors[key]) for key in samples.keys()]
    fig.legend(handles, samples.keys(), bbox_to_anchor=(1.0, 0.5), loc='center left')
    fig.tight_layout()

    return fig, axs

def plot_joint_samples(param_names, colors=None, **samples):
    figsize = (1.2*1.5*len(param_names), 1.5*len(param_names))
    fig, axs = plt.subplots(nrows=len(param_names), ncols=len(param_names), figsize=figsize)

    min = np.min([np.min(s, axis=0) for s in samples.values()], axis=0)
    max = np.max([np.max(s, axis=0) for s in samples.values()], axis=0)

    bins = [np.linspace(min[i], max[i], 31) for i in range(len(param_names))]

    for xi, x_par in enumerate(param_names):
        axs[-1,xi].set_xlabel(x_par, fontsize=28)
        for yi, y_par in enumerate(param_names):
            #axs[yi,xi].xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
            #axs[yi,xi].yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
            axs[yi,xi].set(yticklabels=[], xticklabels=[])
            axs[yi,xi].tick_params(left=False, bottom=False)


            if xi == yi:
                axs[yi,0].set_ylabel(y_par, fontsize=28, rotation=0, labelpad=18)
                for k, s in samples.items():
                    axs[yi,xi].hist(s[:,xi], bins=bins[xi],alpha = 0.5,density=True, color=colors[k])
            elif xi > yi:
                for k, s in samples.items():
                    axs[yi,xi].scatter(s[:,xi], s[:,yi], s=0.5, alpha=0.1, color=colors[k])
            else:
                for k, s in samples.items():
                    axs[yi,xi].scatter(s[:,xi], s[:,yi], s=0.5, alpha=0.1, color=colors[k])


    fig.subplots_adjust(right=0.75)
    handles = [patches.Patch(color=colors[key]) for key in samples.keys()]
    fig.legend(handles, samples.keys(), bbox_to_anchor=(1.2, 0.5), fontsize=20)
    fig.tight_layout()

    return fig, axs

def plot_classification(classification, suptitle, colors, methods = ['Stan', 'BayesFlow']):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs.flatten()
    axs[0].set_title('P(Guessing)', fontsize=14)
    axs[1].set_title('P(Controlled)', fontsize=14)

    x = range(classification.shape[-2])
    for i, method in enumerate(methods):
        for j, s in enumerate(['Guessing', 'Controlled']):
            axs[j].plot(x, classification[1,i,:,j], label = method, color=colors[method])
            axs[j].fill_between(x, classification[0,i,:,j], classification[2,i,:,j], color=colors[method])

    fig.suptitle(suptitle, fontsize=16)
    fig.subplots_adjust(right=0.8)
    handles = [patches.Patch(color=colors[key]) for key in methods]
    fig.legend(handles, methods, bbox_to_anchor=(1.2, 0.5), fontsize=14)
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


def compute_true_log_prob(posterior, stan_model, stan_data):
    log_prob = []

    for iter in range(posterior.shape[0]):
        post = posterior[iter]

        transition_matrix = np.array([
            [post[0], 1-post[0]],
            [1-post[1], post[1]]
        ])

        stan_params = {
            "transition_matrix": transition_matrix,
            "alpha_1": post[2],
            "alpha_2_diff": post[3] - post[2],
            "nu_1": post[4],
            "nu_21": post[5],
            "nu_22_diff": post[6] - post[5],
            "tau": post[7]
        }

        lp = stan_model.log_prob(params = stan_params, data = stan_data, jacobian=False)
        lp = np.array(lp)[0,0]

        log_prob.append(lp)

    log_prob = np.array(log_prob)

    return log_prob
