functions {
    real wald_lpdf(real y, real alpha, real nu) {
        real lpdf;
    
        lpdf = (
            log(alpha) - 1.0/2.0 * log(2*pi()) - 3.0/2.0*log(y) - (alpha - nu*y)^2/(2*y)
        );
    
        return lpdf;
    }

    real wald_rng(real alpha, real nu) {
        real mu = alpha/nu;
        real lambda = alpha^2;
        real zeta = normal_rng(0, 1);
        real zeta_sq = zeta^2;
        real x = mu + (mu^2*zeta_sq)/(2*lambda) - mu/(2*lambda)*sqrt(4*mu*lambda*zeta_sq + mu^2*zeta_sq^2);
        real z = uniform_rng(0, 1);
        real y;
    
        if(z <= mu / (mu + x)){
            y = x;
        } else{
            y = mu^2/x;
        }
    
        return y;
    }

#include hmm/forward.stan
#include hmm/backward.stan
}
data {
    int n_obs; // number of observations (timepoints) to classify
    int n_rep; // number of repetitions per timepoint
    array[n_obs, n_rep] real y; // observations
}
transformed data {
    int n_cls = 2;
    vector[n_cls] log_init_prob = rep_vector(-log(n_cls), n_cls);
}
parameters {
    array[n_cls] simplex[n_cls] transition_matrix;
    real<lower=0> alpha_1; //threshold guessing
    real<lower=0> alpha_2_diff; // diff threshold controlled 
    real<lower=0> nu_1; // drift guessing
    real<lower=0> nu_21; // drift incorrect controlled
    real<lower=0> nu_22_diff; // diff drift correct controlled
    real<lower=0> tau; // non-decision time
}
transformed parameters {
    matrix[n_cls, n_cls] log_transition_matrix;
    array[n_obs] vector[n_cls] emission_log_likelihoods;
    array[n_obs] vector[n_cls] log_alpha; // forward variable
    real alpha_2 = alpha_1 + alpha_2_diff;
    real nu_22 = nu_21 + nu_22_diff;



    for (state in 1:n_cls) {
        log_transition_matrix[state,] = to_row_vector(log(transition_matrix[state]));
    }

    for (obs in 1:n_obs) {
        for (state in 1:n_cls) {
            emission_log_likelihoods[obs, state] = 0.0;
            for (rep in 1:n_rep) {
                if (valid[obs,rep]) {
                    //emission_log_likelihoods[obs, state] += normal_lpdf(y[obs, rep] | mu[state], 1.0);
                }
            }
        }
    }

    log_alpha = forward(n_cls, n_obs, log_init_prob, log_transition_matrix, emission_log_likelihoods);
}
model {
    // priors
    target += dirichlet_lpdf(transition_matrix[1] | [2, 2]);
    target += dirichlet_lpdf(transition_matrix[2] | [2, 2]);
    target += normal_lpdf(alpha_1 | 1.0, 0.5);
    target += normal_lpdf(alpha_2_diff | 0.5, 0.5);
    target += normal_lpdf(nu_1 | 2.0, 0.5);
    target += normal_lpdf(nu_21 | 0.5, 0.5);
    target += normal_lpdf(nu_22_diff | 1.0, 0.5);
}