# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:29:26 2020

@author: alberto.suarez@uam.es
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from numpy.random import default_rng
from typing import Callable, Union
from pandas.plotting import autocorrelation_plot
from tools_qfb import compare_histogram_pdf
from scipy.stats import norm, probplot, t
from scipy.optimize import least_squares, minimize


#%%

def simulate_AR(
    X_0: np.ndarray,
    phi_0: float,
    phi: np.ndarray,
    sigma: float,
    random_number_generator,
    n_trajectories: int,
    n_times: int,
) -> np.ndarray:
    """ Simulation of an AR process

        SDE: X_t = \phi_0 
                   + \sum_{\tau=p} \phi_{\tau} X_{t-\tau} 
                   +  sigma*epsilon_t

    Args:
        times: Integration (monitoring) grid (measurement times).
        X_0: Vector of initial values of the process.
        phi_0: Parameter of the AR process.
        phi: Remaining parameters of the AR process.
        sigma: Noise level.
        n_trajectories: Number of simulated trajectories.
        n_times: Number of values simulated.
        seed: Seed of the random number generator (for reproducibility).

    Returns:
        Simulation consisting of n_trajectories trajectories.
        Each trajectory is a row vector of the values of the simulated process.

    Example:

        >>> phi_0, phi = 0.3, [0.1, -0.8]
        >>> p = len(phi)
        >>> X_0 = np.ones(p) * phi_0 / (1.0 -np.sum(phi))
        >>> sigma = 0.1
        >>> random_number_generator = default_rng(seed=0).standard_normal
        >>> X, u_simulated = simulate_AR(X_0, phi_0, phi, sigma,
        ...    random_number_generator, n_trajectories=50, n_times=1000) 
        >>> fig, ax = plt.subplots() 
        >>> _ = ax.plot(X.T)
        >>> _ = ax.set_xlabel('t')
        >>> _ = ax.set_ylabel('$X_t$') 
        >>> _ = ax.set_title('AR({})'.format(p))
        >>> t_stationary = int(10.0 * t_transient(phi))
        >>> fig, ax = plt.subplots()
        >>> _ = plot_acf(X[0, t_stationary:], lags=30, ax=ax)
        >>> u_computed = residuals_AR(X, phi_0, phi)
        >>> tests_gaussian_white_noise(u_computed[0, t_stationary:])
        >>> print('{:6f}'.format(
        ...     np.max(np.abs(u_simulated[:, p:] - u_computed)))) 
        0.000000
        
        
    """
    p = len(phi)
    X = np.empty((n_trajectories, n_times))
    X[:, :p] = X_0

    epsilon = random_number_generator((n_trajectories, n_times))
    
    u = sigma * epsilon
    
    for t in range(p, n_times):
        X[:, t] = (
            phi_0 
            + X[:, t - np.arange(1, p + 1)] @ phi
            + u[:, t]
        )
        
    return X, u

# %%
def residuals_AR(X, phi_0, phi):
    """ Residuals of an AR model. """
  
    p = len(phi)

    vector = (X.ndim == 1)
    if vector:
        X = X[np.newaxis, :] # time series as a two dimensional array

    u = X[:, p:] - phi_0
    for tau in range(1, p + 1):
        u -=  X[:, (p - tau):-tau] * phi[tau - 1]
    
    if vector:
        return np.ravel(u)  # Return residuals as a one-dimensional array
    else:
        return u


#%%

def simulate_ARMA_GARCH(
    X_0: np.array,
    u_0: np.array,
    h_0: np.ndarray,
    phi_0: float,
    phi: np.ndarray,
    theta: np.ndarray,
    kappa: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    random_number_generator,
    n_trajectories: int,
    n_times: int,
) -> np.ndarray:
    """ Simulation of an ARMA(p, q) + GARCH (r,s) process.

        SDE: X_t = \phi_0 
                   + \sum_{\tau=p} \phi_{\tau} X_{t-\tau} 
                   + \sum_{\tau=p} \theta_{\tau} epsilon_{t-\tau}
                   u_t
                   
             u_t = \sqrt(h_t) *epsilon_t
             h_t = \phi_0 
                   + \sum_{\tau=1^r} \alpha_{\tau} X_{t-\tau} 
                   + \sum_{\tau=p} \theta_{\tau} epsilon_{t-\tau}
                       
    Args:
        times: Integration (monitoring) grid (measurement times).
        X_0: Vector of initial values of the process.
        u_0: Vector of initial values of the innovations.
        phi_0: constant term of the ARMA part of the model.
        phi: AR parameters.
        theta: MA parameters.
        kappa: Constant term of the GARCH part of the model.
        alpha: GARCH coefficients of the square of the delayed innovations.
        beta: GARCH coefficients of the delayed volatities term.
        n_trajectories: Number of simulated trajectories.
        n_times: Number of values simulated.
        seed: Seed of the random number generator (for reproducibility).

    Returns:
        Simulation consisting of n_trajectories trajectories.
        Each trajectory is a row vector of the values of the simulated process.

    Example:

        >>> phi_0, phi= 0.3, [0.1, 0.3]
        >>> p = len(phi) - 1
        >>> theta = [0.3, 0.2]
        >>> q = len(theta)
        >>> kappa = 0.1
        >>> alpha = [0.1, 0.15]
        >>> r = len(alpha) 
        >>> beta = [0.1, 0.6]
        >>> s = len(beta)
        >>> random_number_generator = default_rng(seed=0).standard_normal
        >>> delay = max(p, q, r, s)
        >>> X_0 = np.ones(delay) * phi[0] / (1.0 -np.sum(phi[1:]))
        >>> u_0 = np.zeros(delay)
        >>> h_0 = np.ones(delay) * kappa / (1.0 - np.sum(alpha) - np.sum(beta))
        >>> X, u, h = simulate_ARMA_GARCH(X_0, u_0, h_0, 
        ...    phi_0, phi, theta, kappa, alpha, beta, 
        ...    random_number_generator,
        ...    n_trajectories=50, n_times=1000) 
        >>> fig, ax = plt.subplots() 
        >>> _ = ax.plot(X.T)
        >>> _ = ax.set_xlabel('t')
        >>> _ = ax.set_ylabel('$X_t$') 
        >>> _ = ax.set_title('ARMA({},{}) + GARCH({},{})'.format(p, q, r, s))
        >>> t_stationary = 100
        >>> fig, ax = plt.subplots()    
        >>> _ = plot_acf(X[0, t_stationary:], lags=30, ax=ax)
        >>> tests_gaussian_white_noise(u[0, t_stationary:])
        ... # u is non-Gaussian white noise with nonlinear dependencies.
        >>> epsilon = u[0,t_stationary:] / np.sqrt(h[0,t_stationary:])
        ... # Gaussian white noise (independent because of Gaussianity).        
        >>> tests_gaussian_white_noise(epsilon) 
        
    """
    p = len(phi)
    q = len(theta)
    r = len(alpha)
    s = len(beta)

    delay = max(p, q, r, s)

    X = np.empty((n_trajectories, n_times))
    X[:, :delay] = X_0

    u = np.empty((n_trajectories, n_times))
    u[:, :delay] = u_0

    h = np.empty((n_trajectories, n_times))
    h[:, :delay] = h_0

    epsilon = random_number_generator((n_trajectories, n_times))

    for t in range(delay, n_times):
        h[:, t] = (
            kappa
            + u[:, t - np.arange(1, r + 1)]**2 @ alpha
            + h[:, t - np.arange(1, s + 1)] @ beta
        )

        u[:, t] = np.sqrt(h[:, t]) * epsilon[:, t]
        X[:, t] = (
            phi_0
            + X[:, t - np.arange(1, p + 1)] @ phi
            + u[:, t - np.arange(1, q + 1)] @ theta
            + u[:, t]
        )
        
    return X, u, h

# %%
def residuals_ARMA_GARCH(X, phi_0, phi, theta, kappa, alpha, beta):
    """Residuals of an AR + GARCH model."""

    p = len(phi)
    q = len(theta)
    r = len(alpha)    
    s = len(beta)
    delay = max([p, q, r, s])

    vector = (X.ndim == 1)
    if vector:
        X = X[np.newaxis, :] # time series as a two dimensional array

    n_times = np.shape(X)[1]    

    u = np.empty_like(X)
    h = np.empty_like(X)
    
    # Assume that the initial values of the innovations 
    # are errors of the prediction by the unconditional mean.`
    u[:, :delay] = X[:, :delay] - phi_0 / (1.0 - np.sum(phi)) 

    # Assume that the initial values of the volatility term is the 
    # unconditional variance.
    h[:, :delay] = kappa / (1.0 - np.sum(alpha) - np.sum(beta))
    
    for t in range(delay, n_times):
        u[:, t] = (
            X[:, t] - (
                phi_0
                + X[:, t - np.arange(1, p + 1)] @ phi
                + u[:, t - np.arange(1, q + 1)] @ theta
            )
        )
        
        h[:, t] = (
            kappa
            + u[:, t - np.arange(1, r + 1)]**2 @ alpha
            + h[:, t - np.arange(1, s + 1)] @ beta
        )
        
    if vector:  # Return values as one-dimensional arrays.
        return np.ravel(u), np.ravel(h) 
    else:
        return u, h


# %%

def fit_AR_LS(X, phi_0_seed, phi_seed):
    """ Least-squares error fit to AR process.
    
    Example:
   
        >>> phi_0, phi = 0.3, [0.1, -0.8]
        >>> p = len(phi)
        >>> X_0 = np.ones(p) * phi_0 / (1.0 -np.sum(phi))
        >>> sigma = 0.1
        >>> random_number_generator = default_rng(seed=0).standard_normal
        >>> X, u = simulate_AR(
        ...     X_0, phi_0, phi, sigma,
        ...     random_number_generator, n_trajectories=1, n_times=1000) 
        >>> phi_0_LS, phi_LS, _ = fit_AR_LS(
        ...     np.ravel(X), phi_0_seed=0.0, phi_seed=np.zeros(p))
        >>> print(np.round(phi_0_LS, 1), np.round(phi_LS, 1))
        0.3 [ 0.1 -0.8]
    """

    def mean_squared_error(parameters):
        return np.mean(
            residuals_AR(X, phi_0=parameters[0], phi=parameters[1:])**2
        )
        
    parameters_seed = np.empty(1 + len(phi_seed))
    parameters_seed[0], parameters_seed[1:] = phi_0_seed, phi_seed
    

    info_optimization = least_squares(mean_squared_error, parameters_seed)

    parameters = info_optimization.x  
    
    return parameters[0], parameters[1:], info_optimization

#%% 

def fit_AR_ML_gaussian_noise(X, phi_0_seed, phi_seed, sigma_seed):
    """ Maximum likelihood estimation of an AR model with Gaussian noise."""
    
    from scipy.optimize import minimize
    
    def minus_log_likelihood_AR_gaussian_noise(parameters, X):
        """ Minus the log-likelihood of AR model given the data (Gaussian)."""
        u = residuals_AR(X, phi_0=parameters[0], phi=parameters[1:-1])
        return - np.mean(norm.logpdf(u, loc=0.0, scale=parameters[-1]))

    
    parameters_seed = np.zeros(len(phi_seed) + 2)
    parameters_seed[0] = phi_0_seed
    parameters_seed[1:-1] = phi_seed
    parameters_seed[-1] = sigma_seed
    
    info_optimization = minimize(
        lambda parameters: minus_log_likelihood_AR_gaussian_noise(
            parameters, X
        ),
        parameters_seed,  
        method='nelder-mead',
        options={'xatol': 1e-8, 'disp': False}
    )
    
    parameters = info_optimization.x  

    phi_0 = parameters[0]
    phi = parameters[1:-1]
    sigma = parameters[-1]
    return phi_0, phi, sigma, info_optimization
#%%
def fit_AR_ML_student_t_noise(X, phi_0_seed, phi_seed, sigma_seed, nu_seed):
    # Vector inicial
    theta0 = np.array([phi_0_seed, phi_seed[0], sigma_seed, nu_seed])
    
    def neg_log_likelihood(theta):
        phi_0, phi_1, sigma, nu = theta
        
        # Restricciones de viabilidad matemática
        if sigma <= 0 or nu <= 2.0: 
            return 1e10
            
        # Residuos del AR(1) vectorizados para mayor velocidad
        u = X[1:] - (phi_0 + phi_1 * X[:-1])
        
        # Log-Verosimilitud
        log_L = np.sum(t.logpdf(u, df=nu, loc=0.0, scale=sigma))
        return -log_L

    # Límites
    bnds = (
        (None, None),    # phi_0
        (-0.999, 0.999), # phi_1
        (1e-6, None),    # sigma > 0
        (2.001, 100.0)   # nu > 2
    )
    
    info_optimization = minimize(neg_log_likelihood, theta0, method='SLSQP', bounds=bnds)
    
    parameters = info_optimization.x
    phi_0 = parameters[0]
    phi = np.array([parameters[1]])
    sigma = parameters[2]
    nu = parameters[3]
    
    return phi_0, phi, sigma, nu, info_optimization



def fit_AR_GARCH_ML_gaussian_noise(
    log_returns, phi_0_seed, phi_seed, kappa_seed, alpha_seed, beta_seed, 
    method='SLSQP', apply_constraints=True
):
    
    # Vector inicial
    theta0 = np.array([phi_0_seed, phi_seed[0], kappa_seed, alpha_seed[0], beta_seed[0]])
    
    def neg_log_likelihood(theta):
        phi_0, phi_1, kappa, alpha, beta = theta
        T = len(log_returns)
        
        u = np.zeros(T)
        h = np.zeros(T)
        
        u[0] = log_returns[0] - phi_0
        h[0] = np.var(log_returns)
        
        log_L = 0.0
        
        for t_idx in range(1, T):
            u[t_idx] = log_returns[t_idx] - (phi_0 + phi_1 * log_returns[t_idx-1])
            h[t_idx] = kappa + alpha * (u[t_idx-1]**2) + beta * h[t_idx-1]
            
            # Barrera de emergencia (Soft constraint) crucial si se usa Nelder-Mead
            if h[t_idx] <= 0:
                return 1e10
                
            log_L += norm.logpdf(u[t_idx], loc=0.0, scale=np.sqrt(h[t_idx]))
            
        return -log_L

    # Definición formal de límites y restricciones
    bnds = (
        (None, None),        # phi_0: sin límite
        (-0.999, 0.999),     # phi_1: estacionariedad del AR(1)
        (1e-6, None),        # kappa > 0
        (1e-6, 0.999),       # alpha > 0
        (1e-6, 0.999)        # beta > 0
    )
    # 1 - alpha - beta >= 0 (Estacionariedad en covarianza)
    cons = ({'type': 'ineq', 'fun': lambda theta: 0.999 - theta[3] - theta[4]})

    # Configuración dinámica del optimizador
    opt_kwargs = {'method': method, 'options': {'maxiter': 1000}}
    
    if apply_constraints:
        # Los límites funcionan en SLSQP, L-BFGS-B, TNC, trust-constr
        if method in ['SLSQP', 'L-BFGS-B', 'TNC', 'trust-constr']:
            opt_kwargs['bounds'] = bnds
        # Las inecuaciones funcionan en SLSQP, COBYLA, trust-constr
        if method in ['SLSQP', 'COBYLA', 'trust-constr']:
            opt_kwargs['constraints'] = cons

    # Optimización
    res = minimize(neg_log_likelihood, theta0, **opt_kwargs)
    
    phi_0_opt = res.x[0]
    phi_1_opt = np.array([res.x[1]])
    kappa_opt = res.x[2]
    alpha_opt = np.array([res.x[3]])
    beta_opt = np.array([res.x[4]])
    
    return phi_0_opt, phi_1_opt, kappa_opt, alpha_opt, beta_opt, res


def fit_AR_GARCH_ML_student_t_noise(
    log_returns, phi_0_seed, phi_seed, kappa_seed, alpha_seed, beta_seed, nu_seed, 
    method='SLSQP', apply_constraints=True
):
    
    # Vector inicial (añadimos nu)
    theta0 = np.array([phi_0_seed, phi_seed[0], kappa_seed, alpha_seed[0], beta_seed[0], nu_seed])
    
    def neg_log_likelihood(theta):
        phi_0, phi_1, kappa, alpha, beta, nu = theta
        T = len(log_returns)
        
        u = np.zeros(T)
        h = np.zeros(T)
        
        u[0] = log_returns[0] - phi_0
        h[0] = np.var(log_returns)
        
        log_L = 0.0
        
        for t_idx in range(1, T):
            u[t_idx] = log_returns[t_idx] - (phi_0 + phi_1 * log_returns[t_idx-1])
            h[t_idx] = kappa + alpha * (u[t_idx-1]**2) + beta * h[t_idx-1]
            
            if h[t_idx] <= 0:
                return 1e10
                
            log_L += t.logpdf(u[t_idx], df=nu, loc=0.0, scale=np.sqrt(h[t_idx]))
            
        return -log_L

    # Definición formal de límites y restricciones
    bnds = (
        (None, None),        # phi_0
        (-0.999, 0.999),     # phi_1
        (1e-6, None),        # kappa > 0
        (1e-6, 0.999),       # alpha > 0
        (1e-6, 0.999),       # beta > 0
        (2.001, 100.0)       # nu > 2 (Varianza definida para t-Student)
    )
    cons = ({'type': 'ineq', 'fun': lambda theta: 0.999 - theta[3] - theta[4]})

    # Configuración dinámica del optimizador
    opt_kwargs = {'method': method, 'options': {'maxiter': 1000}}
    
    if apply_constraints:
        if method in ['SLSQP', 'L-BFGS-B', 'TNC', 'trust-constr']:
            opt_kwargs['bounds'] = bnds
        if method in ['SLSQP', 'COBYLA', 'trust-constr']:
            opt_kwargs['constraints'] = cons

    # Optimización
    res = minimize(neg_log_likelihood, theta0, **opt_kwargs)
    
    phi_0_opt = res.x[0]
    phi_1_opt = np.array([res.x[1]])
    kappa_opt = res.x[2]
    alpha_opt = np.array([res.x[3]])
    beta_opt = np.array([res.x[4]])
    nu_opt = res.x[5]
    
    return phi_0_opt, phi_1_opt, kappa_opt, alpha_opt, beta_opt, nu_opt, res




def fit_AR_GARCH_ML_gaussian_noise(
    log_returns, phi_0_seed, phi_seed, kappa_seed, alpha_seed, beta_seed, 
    method='SLSQP', apply_constraints=True
):
    
    # Vector inicial
    theta0 = np.array([phi_0_seed, phi_seed[0], kappa_seed, alpha_seed[0], beta_seed[0]])
    
    def neg_log_likelihood(theta):
        phi_0, phi_1, kappa, alpha, beta = theta
        T = len(log_returns)
        
        u = np.zeros(T)
        h = np.zeros(T)
        
        u[0] = log_returns[0] - phi_0
        h[0] = np.var(log_returns)
        
        log_L = 0.0
        
        for t_idx in range(1, T):
            u[t_idx] = log_returns[t_idx] - (phi_0 + phi_1 * log_returns[t_idx-1])
            h[t_idx] = kappa + alpha * (u[t_idx-1]**2) + beta * h[t_idx-1]
            
            # Barrera de emergencia (Soft constraint) crucial si se usa Nelder-Mead
            if h[t_idx] <= 0:
                return 1e10
                
            log_L += norm.logpdf(u[t_idx], loc=0.0, scale=np.sqrt(h[t_idx]))
            
        return -log_L

    # Definición formal de límites y restricciones
    bnds = (
        (None, None),        # phi_0: sin límite
        (-0.999, 0.999),     # phi_1: estacionariedad del AR(1)
        (1e-6, None),        # kappa > 0
        (1e-6, 0.999),       # alpha > 0
        (1e-6, 0.999)        # beta > 0
    )
    # 1 - alpha - beta >= 0 (Estacionariedad en covarianza)
    cons = ({'type': 'ineq', 'fun': lambda theta: 0.999 - theta[3] - theta[4]})

    # Configuración dinámica del optimizador
    opt_kwargs = {'method': method, 'options': {'maxiter': 1000}}
    
    if apply_constraints:
        # Los límites funcionan en SLSQP, L-BFGS-B, TNC, trust-constr
        if method in ['SLSQP', 'L-BFGS-B', 'TNC', 'trust-constr']:
            opt_kwargs['bounds'] = bnds
        # Las inecuaciones funcionan en SLSQP, COBYLA, trust-constr
        if method in ['SLSQP', 'COBYLA', 'trust-constr']:
            opt_kwargs['constraints'] = cons

    # Optimización
    res = minimize(neg_log_likelihood, theta0, **opt_kwargs)
    
    phi_0_opt = res.x[0]
    phi_1_opt = np.array([res.x[1]])
    kappa_opt = res.x[2]
    alpha_opt = np.array([res.x[3]])
    beta_opt = np.array([res.x[4]])
    
    return phi_0_opt, phi_1_opt, kappa_opt, alpha_opt, beta_opt, res


#%%
def t_transient(phi):
    """ Compute the length of the transient regime for an AR(p) process."""
    
    p = len(phi)
    phi_matrix = np.diag(np.ones(p - 1), -1)
    phi_matrix[0, :] = phi
    eigenvalue_max_abs_value = np.max(np.abs(np.linalg.eig(phi_matrix)[0]))
    
    return - 1.0 / np.log(eigenvalue_max_abs_value)
                                     

# %%

def tests_gaussian_white_noise(noise, figsize=(12, 4)):
    """ Tests to determine whether sample is Gaussian white noise.
     
    """
    
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True)   
    
    mu, sigma = 0.0, np.std(noise)
    
    compare_histogram_pdf(
        noise, 
        lambda x: norm.pdf(x, mu, sigma),
        ax=axs[0],
    )
    
    probplot(noise, sparams=(mu, sigma), dist='norm', plot=axs[1])
    
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    _ = plot_acf(noise, lags=30, ax=axs[0]) # linear autocorrelations
    _ = plot_acf(np.abs(noise), lags=30, ax=axs[1]) # non-linear dependencies 
    _ = axs[0].set_title(
        'Autocorrelations of the time series.')
    _ = axs[0].set_xlabel(r'$\tau$')
    _ = axs[0].set_ylabel(r'$\rho(\tau)$')
    
    _ = axs[1].set_title(
        'Autocorrelations of the absolute values of the time series.')
    _ = axs[1].set_xlabel(r'$\tau$')
    _ = axs[1].set_ylabel(r'$\rho(\tau)$')
    
#%%

# Run examples and test results


if __name__ == "__main__":
    import doctest
    doctest.testmod()
