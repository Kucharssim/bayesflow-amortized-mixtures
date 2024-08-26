functions {
    real wald_lpdf(real y, real alpha, real nu) {
        real lpdf;
    
        lpdf = (
            log(alpha) - 1.0/2.0 * log(2*pi()) - 3.0/2.0*log(y) - (alpha - nu*y)^2/(2*y)
        );
    
        return lpdf;
    }

    real wald_lcdf(real y, real alpha, real nu) {
        real mu = alpha/nu;
        real lambda = alpha^2;

        real ly = sqrt(lambda/y);
        real ymu = y/mu;

        vector[2] terms;
        
        terms[1] = std_normal_lcdf(ly * (ymu - 1));
        terms[2] = 2*lambda/mu;
        terms[2] += std_normal_lcdf(- ly * (ymu + 1));

        real result = log_sum_exp(terms);

        return result;
    }

    real wald_lccdf(real y, real alpha, real nu) {
        real lcdf = wald_lcdf(y | alpha, nu);
        return log1m_exp(lcdf);
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
        } else {
            y = mu^2/x;
        }
    
        return y;
    }

    real rdm_lpdf(tuple(int, real) y, int k, vector alpha, vector nu, vector tau) {
        int choice = y.1;
        real rt = y.2;
        real lpdf = 0.0;

        for (i in 1:k) {
            if (i == choice) {
                lpdf += wald_lpdf(rt - tau[i] | alpha[i], nu[i]);
            } else {
                lpdf += wald_lccdf(rt - tau[i] | alpha[i], nu[i]);
            }
        }
        return lpdf;
    }

#include hmm/forward.stan
#include hmm/backward.stan
}
data {
    int n_obs; // number of observations (timepoints) to classify
    array[n_obs] tuple (int, real) y; // observations
}
transformed data {
    int n_cls = 2;
    vector[n_cls] log_init_prob = rep_vector(-log(n_cls), n_cls);
    vector[n_obs] rts; 
    for (i in 1:n_obs) {
        rts[i] = y[i].2;
    }
}
parameters {
    array[n_cls] simplex[n_cls] transition_matrix;
    real<lower=0> alpha_1; //threshold guessing
    real<lower=0> alpha_2_diff; // diff threshold controlled 
    real<lower=0> nu_1; // drift guessing
    real<lower=0> nu_21; // drift incorrect controlled
    real<lower=0> nu_22_diff; // diff drift correct controlled
    real<lower=0,upper=min(rts)> tau; // non-decision time
}
transformed parameters {
    matrix[n_cls, n_cls] log_transition_matrix;
    array[n_obs] vector[n_cls] emission_log_likelihoods;
    array[n_obs] vector[n_cls] log_alpha; // forward variable
    vector[2] nu_2;
    real alpha_2 = alpha_1 + alpha_2_diff;

    nu_2[1] = nu_21;
    nu_2[2] = nu_21 + nu_22_diff;



    for (state in 1:n_cls) {
        log_transition_matrix[state,] = to_row_vector(log(transition_matrix[state]));
    }

    for (obs in 1:n_obs) {
        emission_log_likelihoods[obs,1] = wald_lpdf(y[obs].2 - tau | alpha_1, nu_1) - log(2.0);
        emission_log_likelihoods[obs,2] = rdm_lpdf(y[obs] | 2, rep_vector(alpha_2, 2), nu_2, rep_vector(tau, 2));
    }

    log_alpha = forward(n_cls, n_obs, log_init_prob, log_transition_matrix, emission_log_likelihoods);
}
model {
    // likelihood
     target += log_sum_exp(log_alpha[n_obs]);

    // priors
    target += dirichlet_lpdf(transition_matrix[1] | [16, 4]);
    target += dirichlet_lpdf(transition_matrix[2] | [4, 16]);
    target += normal_lpdf(alpha_1 | 0.5, 0.3);
    target += normal_lpdf(alpha_2_diff | 1.5, 0.5);
    target += normal_lpdf(nu_1 | 5.5, 1.0);
    target += normal_lpdf(nu_21 | 2.5, 0.5);
    target += normal_lpdf(nu_22_diff | 2.5, 1.0);
    target += exponential_lpdf(tau | 5.0);
}
generated quantities {
    array[n_obs] vector[n_cls] log_beta; //backward variable
    array[n_obs] simplex[n_cls] smoothing; // p(state_t | y_{1:T})
    array[n_obs] simplex[n_cls] filtering; // p(state_t | y_{1:t})
    array[n_obs] simplex[n_cls] backward_filtering; // p(state_t | y_{t+1:T})

    log_beta = backward(n_cls, n_obs, log_init_prob, log_transition_matrix, emission_log_likelihoods);

    for (obs in 1:n_obs) {
        smoothing[obs]          = softmax(log_alpha[obs] + log_beta[obs]);
        filtering[obs]          = softmax(log_alpha[obs]);
        backward_filtering[obs] = softmax(log_beta[obs]);
    }
}