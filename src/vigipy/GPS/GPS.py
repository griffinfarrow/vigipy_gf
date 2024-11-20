import pandas as pd
import numpy as np
import warnings
from scipy.special import gdtr
from scipy.special import digamma
from scipy.optimize import minimize, dual_annealing
from scipy.stats import nbinom 

from ..utils import Container
from ..utils import calculate_expected
from ..utils.distribution_funcs.negative_binomials import dnbinom, pnbinom
from ..utils.distribution_funcs.quantile_funcs import quantiles
from .callback_functions import Callback, Anneal_Callback

dnbinom = np.vectorize(dnbinom)
pnbinom = np.vectorize(pnbinom)
quantiles = np.vectorize(quantiles)

EPS = np.finfo(np.float32).eps
BOUNDED_METHODS = {
    "Nelder-Mead",
    "L-BFGS-B",
    "TNC",
    "SLSQP",
    "Powell",
    "trust-constr",
    "COBYLA",
    "COBYQA",
    "DA" # Dual annealing
}

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
class gps:
    """
    Perform disproportionality analysis using the Multi-Item Gamma Poisson Shrinker (GPS) algorithm.

    This class implements a gamma-poisson shrinker algorithm for analyzing adverse event (AE) 
    data and detecting disproportionality signals. It optimizes hyperparameters, calculates posterior 
    probabilities, and generates ranking statistics based on specified metrics.

    Attributes:
        container (DataContainer): 
            A DataContainer object produced by the `convert()` function from `data_prep.py`, 
            containing the input data for analysis.
        
        relative_risk (float, optional): 
            The relative risk value used in the analysis (default is 1).
        
        min_events (int, optional): 
            The minimum number of adverse event (AE) reports required to be considered a signal 
            (default is 1).
        
        decision_metric (str, optional): 
            The metric used for detecting signals. Options include:
            - 'fdr': False Detection Rate
            - 'signals': Number of Signals
            - 'rank': Ranking Statistic
            (default is 'rank').
        
        decision_thres (float, optional): 
            The threshold value used for the decision_metric to determine signal significance 
            (default is 0.05).
        
        ranking_statistic (str, optional): 
            The statistic used to rank signals. Options include:
            - 'p_value': Posterior Probability
            - 'quantile': 5% Quantile of the Lambda Distribution
            - 'log2': Posterior Expectation of log2(lambda)
            (default is 'log2').
        
        truncate (bool, optional): 
            Whether to apply truncation based on a minimum number of notifications for hyperparameter 
            calculations (default is False).
        
        truncate_thres (int, optional): 
            The threshold number of notifications for truncation if truncate is True (default is 1).
        
        prior_init (dict, optional): 
            Initial priors for the multi-item gamma poisson shrinkage. Defaults are based on 
            DuMouchel's 1999 paper (default values provided).
        
        prior_param (array-like, optional): 
            Specific hyperparameters to use for the analysis. If None, hyperparameters will be 
            optimized (default is None).
        
        expected_method (str, optional): 
            Method used for calculating expected counts for disproportionality analysis. Options 
            include 'mantel-haentzel' (default).
        
        method_alpha (float, optional): 
            If `expected_method` is 'negative-binomial', this parameter specifies the alpha parameter 
            of the distribution (default is 1).
        
        minimization_method (str, optional): 
            Optimization method for `scipy.optimize.minimize()` (default is 'CG').
        
        minimization_bounds (tuple, optional): 
            Bounds for the optimization parameters (default is a tuple with specific bounds).
        
        minimization_options (dict, optional): 
            Additional options for `scipy.optimize.minimize()` (default is None).
        
        message (bool, optional): 
            Whether to print convergence messages during optimization (default is False).
        
        opt_likelihood (bool, optional): 
            Whether to use the optimized likelihood function (default is False). This can be faster 
            but is experimental.
        
        sim_anneal (bool, optional): 
            Whether to use simulated annealing for optimization (default is False). This is an 
            experimental feature.
        
        number_of_iterations (int, optional): 
            Number of iterations to use in the optimization process (default is 1000).
        
        tol_value (float, optional): 
            Tolerance value for the optimization function (default is 1.0e-4).
        
        product_label (str, optional): 
            The column name in the DataContainer for product data (default is 'name').
        
        ae_label (str, optional): 
            The column name in the DataContainer for adverse event data (default is 'AE').
            
        squashing (bool, optional):
            Whether to optimise using squashed data (default is False)
            
        callback_freq (int, optional):
            How many iterations to return a callback of function value after

    Methods:
        workflow: 
            Runs the full analysis workflow including data preparation, hyperparameter optimization, 
            EBGM score calculation, signal report generation, and results output.
    """
    def __init__(
        self,
        container,
        relative_risk=1,
        min_events=1,
        decision_metric="rank",
        decision_thres=0.05,
        ranking_statistic="log2",
        truncate=False,
        truncate_thres=1,
        prior_init=None,
        prior_param=None,
        expected_method="mantel-haentzel",
        method_alpha=1,
        minimization_method="CG",
        minimization_bounds=None,
        minimization_options=None,
        message=False,
        opt_likelihood=False,
        number_of_iterations=1000,
        tol_value=1.0e-4,
        sim_anneal=False,
        product_label='name',
        ae_label='AE',
        squashing=False,
        callback_freq=50
    ):
        # Set default prior_init if none provided
        if prior_init is None:
            prior_init = {
                "alpha1": 0.2041,
                "beta1": 0.05816,
                "alpha2": 1.415,
                "beta2": 1.838,
                "w": 0.0969,
            }
        
        # Set default minimization_bounds if none provided
        if minimization_bounds is None:
            minimization_bounds = (
                (EPS, 20), 
                (EPS, 10), 
                (EPS, 20), 
                (EPS, 10), 
                (EPS, 1 - EPS)
            )
        
        # Initialize object attributes
        self.container = container 
        self.relative_risk = relative_risk 
        self.min_events = min_events 
        self.decision_metric = decision_metric 
        self.decision_thres = decision_thres
        self.ranking_statistic = ranking_statistic
        self.truncate = truncate
        self.truncate_thres = truncate_thres
        self.prior_init = prior_init
        self.prior_param = prior_param
        self.expected_method = expected_method 
        self.method_alpha = method_alpha
        self.minimization_method = minimization_method 
        self.minimization_bounds = minimization_bounds
        self.minimization_options = minimization_options 
        self.message = message
        self.opt_likelihood = opt_likelihood 
        self.number_of_iterations = number_of_iterations
        self.tol_value = tol_value 
        self.sim_anneal = sim_anneal 
        self.product_label = product_label 
        self.ae_label = ae_label 
        self.squashing = squashing
        self.callback_freq = callback_freq

    def initial_data_prep(self):
        """Initial data preparation and prior initialization."""
        if (self.message):
            print("BEGINNING INITIAL DATA PREP")
        
        # Convert priors to a numpy array
        self.priors = np.asarray(
            [
                self.prior_init["alpha1"],
                self.prior_init["beta1"],
                self.prior_init["alpha2"],
                self.prior_init["beta2"],
                self.prior_init["w"],
            ]
        )

        # Extract data from container
        self.DATA = self.container.data
        self.N = self.container.N

        # Prepare relevant arrays for calculations
        self.n11 = np.asarray(self.DATA["events"], dtype=np.float64)
        self.n1j = np.asarray(self.DATA["product_aes"], dtype=np.float64)
        self.ni1 = np.asarray(self.DATA["count_across_brands"], dtype=np.float64)

        # Calculate expected counts based on the selected method
        self.expected = calculate_expected(
            self.N, self.n1j, self.ni1, self.n11, 
            self.expected_method, self.method_alpha
        )

        self.p_out = True
        
        if (self.squashing):
            print("You are using squashing: make sure you now call initial_data_prep_squashing")
            
    def initial_data_prep_squashing(self, counts_squashed, expected_squashed, weights_squashed):
        """Saves squashed data so that it can be used for optimisation"""
        self.n_squashed = counts_squashed
        self.e_squashed = expected_squashed
        self.weights_squashed = weights_squashed

    def optimize(self):
        """Hyperparameter optimization for priors."""
        if self.prior_param is None:
            if (self.message):
                print("BEGINNING HYPERPARAMETER OPTIMISATION")
                
            if (not self.squashing):
                # If prior_param is not provided, perform optimization
                self.p_out = hyperparameter_optimization(
                    self.container, self.truncate, self.truncate_thres, 
                    self.priors, self.n11, self.expected, 
                    self.minimization_method, self.minimization_bounds, 
                    self.minimization_options, self.message, 
                    self.opt_likelihood, self.number_of_iterations, 
                    self.tol_value, self.sim_anneal, self.callback_freq
                )
            else:
                print("YOU ARE USING SQUASHING: USE WITH CAUTION")
                self.p_out = hyperparameter_opt_squash(
                    self.truncate, self.truncate_thres, self.priors, 
                    self.n_squashed, self.e_squashed, self.weights_squashed,
                    self.minimization_method, self.minimization_bounds,
                    self.minimization_options, self.message, 
                    self.number_of_iterations, self.tol_value, self.sim_anneal,
                    self.callback_freq
                )
            
            # Update priors with the optimized values
            self.priors = self.p_out.x

            # Check if priors meet constraints, else warn user
            if np.any(self.priors < 0) or self.priors[4] > 1:
                warnings.warn(
                    f"Calculated priors violate distribution constraints. "
                    f"Alpha and Beta parameters should be >0 and mixture weight should be >=0 and <=1. "
                    f"Current priors: {self.priors}. Numerical instability likely."
                )

            # Store convergence information
            self.code_convergence = self.p_out.message

            if self.message:
                print("="*50)
                print("OPTIMISED PRIORS REACHED: ", self.priors)
                print("OPTIMISED FUNCTION VALUE = ", self.p_out.fun)
                print("="*50)
                print(self.code_convergence)

        else:
            if self.message:
                print(f"USING PROVIDED PRIORS {self.prior_param}")
            self.priors = self.prior_param

        # Diagnostic likelihood evaluation with optimized priors
        self.likeli = local_likelihood_diagnostic(
            self.container, self.truncate, self.truncate_thres, 
            self.n11, self.expected, self.priors, self.N
        )

        if np.isnan(self.likeli):
                warnings.warn(
                    "Calculated likelihood is NaN. Implies an error somewhere!"
                    "This will likely cause the report generation part of the code to fail"
                )


        # Apply minimum event filtering if needed
        if self.min_events > 1:
            valid_indices = self.n11 >= self.min_events
            self.DATA = self.DATA[valid_indices]
            self.expected = self.expected[valid_indices]
            self.n1j = self.n1j[valid_indices]
            self.ni1 = self.ni1[valid_indices]
            self.n11 = self.n11[valid_indices]

        self.num_cell = len(self.n11)

    def ebgm_calculate(self):
        """Calculate EBGM scores and quantiles."""
        if self.message:
            print("CALCULATING EBGM SCORES")

        # Posterior probability of the null hypothesis
        self.Qn, self.EBlog2, self.posterior_probability = ebgm_calculation(
            self.priors, self.expected, self.n11, self.relative_risk
        )

        if self.message:
            print("CALCULATING QUANTILES")

        # Calculate the lower bound of the quantile distribution
        self.LB = quantiles(
            0.05, self.Qn, 
            self.priors[0] + self.n11, self.priors[1] + self.expected, 
            self.priors[2] + self.n11, self.priors[3] + self.expected
        )

    def report_generate(self):
        """Generate ranking statistics for signal report."""
        if self.message:
            print("GENERATING REPORT")
        
        # Generate signal-related metrics like FDR, FNR, Se, Sp, and rank statistic
        self.FDR, self.FNR, self.Se, self.Sp, self.RankStat, self.num_signals = generate_ranking_stats(
            self.ranking_statistic, self.EBlog2, self.LB, 
            self.posterior_probability, self.num_cell, 
            self.decision_metric, self.decision_thres
        )

    def output_results(self):
        """Generate and return the final signal report."""
        if self.message:
            print("PRODUCING FINAL REPORT")

        # Prepare data columns for the final report
        name = self.DATA[self.product_label]
        ae = self.DATA[self.ae_label]
        count = self.n11

        RES = Container(params=True)

        # Store parameters and diagnostic information
        RES.param = {
            "input_params": {
                "relative_risk": self.relative_risk,
                "min_events": self.min_events,
                "decision_metric": self.decision_metric,
                "decision_thres": self.decision_thres,
            },
            "prior_init": self.prior_init,
            "prior_param": self.priors,
            "likelihood_at_min": self.likeli
        }

        if self.prior_param is None:
            RES.param["convergence"] = self.code_convergence

        # Depending on the ranking statistic, create the report in the appropriate format
        if self.ranking_statistic == "p_value":
            RES.all_signals = pd.DataFrame({
                "Product": name,
                "Adverse Event": ae,
                "Count": count,
                "Expected Count": self.expected,
                "p_value": self.RankStat,
                "count/expected": (self.n11 / self.expected),
                "product margin": self.n1j,
                "event margin": self.ni1,
                "fdr": self.FDR,
                "FNR": self.FNR,
                "Se": self.Se,
                "Sp": self.Sp,
            }).sort_values(by=[self.ranking_statistic])

        elif self.ranking_statistic == "quantile":
            RES.all_signals = pd.DataFrame({
                "Product": name,
                "Adverse Event": ae,
                "Count": count,
                "Expected Count": self.expected,
                "quantile": self.RankStat,
                "count/expected": (self.n11 / self.expected),
                "product margin": self.n1j,
                "event margin": self.ni1,
                "fdr": self.FDR,
                "FNR": self.FNR,
                "Se": self.Se,
                "Sp": self.Sp,
                "posterior_probability": self.posterior_probability,
            }).sort_values(by=[self.ranking_statistic], ascending=False)

        else:
            RES.all_signals = pd.DataFrame({
                "Product": name,
                "Adverse Event": ae,
                "Count": count,
                "Expected Count": self.expected,
                "log2": self.RankStat,
                "count/expected": (self.n11 / self.expected),
                "product margin": self.n1j,
                "event margin": self.ni1,
                "fdr": self.FDR,
                "FNR": self.FNR,
                "Se": self.Se,
                "Sp": self.Sp,
                "LowerBound": self.LB,
                "p_value": self.posterior_probability,
            }).sort_values(by=[self.ranking_statistic], ascending=False)

        # Update signal indices and limit signals if required
        RES.all_signals.index = np.arange(len(RES.all_signals.index))

        if self.num_signals > 0:
            self.num_signals -= 1
        else:
            self.num_signals = 0

        RES.signals = RES.all_signals.iloc[:self.num_signals]
        RES.num_signals = self.num_signals

        return RES

    def workflow(self):
        """Run the full workflow: from data prep to final results."""
        self.initial_data_prep()
        
        self.optimize()
        if np.isnan(self.likeli):
            raise Exception("Likelihood is a NaN, EBGM calculation will hang, correct!")
        
        self.ebgm_calculate()
        self.report_generate()
        results = self.output_results()
        
        return results

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def non_truncated_likelihood(p, n11, E):
    """
    Compute the non-truncated likelihood using an non-optimized approach.

    This uses custom functions to calculate the different components. Would not recommend using this
    as it is a factor 50x slower and gives identical answers to the optimized approach

    Parameters:
        p: array-like
            Parameters for the likelihood functions. These come from the hyperparameter optimisation
            The array should contain:
            - p[0]: Size parameter for the first negative binomial distribution.
            - p[1]: Success parameter for the first negative binomial distribution.
            - p[2]: Size parameter for the second negative binomial distribution.
            - p[3]: Success parameter for the second negative binomial distribution.
            - p[4]: Mixture proportion parameter.

        n11: array-like
            Observed frequency data for which the likelihood is computed.

        E: array-like
            Expected frequency data used for calculating the probabilities.

    Returns:
        float
            The computed negative log-likelihood. The result is obtained by summing the 
            negative logarithm of the combined likelihood values, with a small constant 
            added to prevent log(0) issues.

    Notes:
        - The probabilities are calculated for two negative binomial distributions using 
          the provided parameters.
        - The likelihood is computed as a weighted sum of these probabilities, where the 
          weights are determined by the mixture proportion parameter.
        - A small constant (1e-7) is added to the combined likelihood to avoid taking 
          the logarithm of zero, which could result in numerical instability.
        - NaN values in the probability calculations are handled by converting them to zero 
          using `np.nan_to_num`.
    """
    dnb1 = np.nan_to_num(dnbinom(n11, prob=p[1] / (p[1] + E), size=p[0]))
    dnb2 = np.nan_to_num(dnbinom(n11, prob=p[3] / (p[3] + E), size=p[2]))
    term = (p[4] * dnb1 + (1 - p[4]) * dnb2) + 1e-7
    return np.sum(-np.log(term))

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def non_truncated_likelihood_optimised(p, n11, E):
    """
    Compute the non-truncated likelihood using an optimized approach.

    This function calculates the likelihood based on the negative binomial distribution 
    without applying any truncation. It uses vectorized operations to efficiently compute 
    probabilities and combine them based on a mixture model.

    Parameters:
        p: array-like
            Parameters for the likelihood functions. These come from the hyperparameter optimisation
            The array should contain:
            - p[0]: Size parameter for the first negative binomial distribution.
            - p[1]: Success parameter for the first negative binomial distribution.
            - p[2]: Size parameter for the second negative binomial distribution.
            - p[3]: Success parameter for the second negative binomial distribution.
            - p[4]: Mixture proportion parameter.

        n11: array-like
            Observed frequency data for which the likelihood is computed.

        E: array-like
            Expected frequency data used for calculating the probabilities.

    Returns:
        float
            The computed negative log-likelihood. The result is obtained by summing the 
            negative logarithm of the combined likelihood values, with a small constant 
            added to prevent log(0) issues.

    Notes:
        - The probabilities are calculated for two negative binomial distributions using 
          the provided parameters.
        - The likelihood is computed as a weighted sum of these probabilities, where the 
          weights are determined by the mixture proportion parameter.
        - A small constant (1e-7) is added to the combined likelihood to avoid taking 
          the logarithm of zero, which could result in numerical instability.
        - NaN values in the probability calculations are handled by converting them to zero 
          using `np.nan_to_num`.
    """
    # Vectorized calculation of probabilities
    prob1 = p[1] / (p[1] + E)
    prob2 = p[3] / (p[3] + E)
    # Vectorized calculation of negative binomial distributions
    dnb1 = np.nan_to_num(nbinom.pmf(n11, p[0], prob1))
    dnb2 = np.nan_to_num(nbinom.pmf(n11, p[2], prob2))
    # Combine terms
    term = (p[4] * dnb1 + (1 - p[4]) * dnb2) + 1e-7
    term = np.clip(term, 1.0e-10, None) # to prevent log(0) errors
    # Return the sum of the negative log-likelihood
    return np.sum(-np.log(term))

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def truncated_likelihood(p, n11, E, truncate):
    """
    Compute the truncated likelihood using an optimized approach with two versions.

    This function calculates the likelihood based on a truncated version of the 
    negative binomial distribution. 

    N.B.: This is the version of the code that originally shipped with the Vigipy code
    We are fairly sure this produces completely incorrect answers (and slowly)
    DO NOT use this function unless you are very sure what you are doing 
    THIS NEEDS A THOROUGH DEBUGGING FIRST

    See truncated_likelihood_optimised for a description of the arguments and returned valuesw
    """
    warnings.warn(
        "THIS FUNCTION HAS NEVER BEEN DEBUGGED AND DIDN'T WORK AS OF SEPT 2024"
        "DO NOT USE UNLESS YOU ARE VERY SURE WHAT YOU ARE DOING"
    )
    
    dnb1 = dnbinom(n11, size=p[0], prob=p[1] / (p[1] + E))
    dnb2 = dnbinom(n11, size=p[2], prob=p[3] / (p[3] + E))
    term1 = p[4] * dnb1 + (1 - p[4]) * dnb2
    pnb1 = pnbinom(truncate, size=p[0], prob=p[1] / (p[1] + E))
    pnb2 = pnbinom(truncate, size=p[2], prob=p[3] / (p[3] + E))
    term2 = 1 - (p[4] * pnb1 + (1 - p[4]) * pnb2)
    return np.sum(-np.log(term1 / term2))

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def truncated_likelihood_optimised(p, n11, E, truncate):
    """
    Compute the truncated likelihood using an optimized approach with two versions.

    This function calculates the likelihood based on a truncated version of the 
    negative binomial distribution. Two versions are provided: one that is designed 
    to ensure compatibility with R/OpenEBGM and another that is the original Python/Vigipy 
    implementation. 

    Parameters:
        p: array-like
            Parameters for the likelihood functions. These come from the hyperparameter optimisation
            The array should contain:
            - p[0]: Size parameter for the first negative binomial distribution.
            - p[1]: Success parameter for the first negative binomial distribution.
            - p[2]: Size parameter for the second negative binomial distribution.
            - p[3]: Success parameter for the second negative binomial distribution.
            - p[4]: Mixture proportion parameter.

        n11: array-like
            Observed frequency data for which the likelihood is computed.

        E: array-like
            Expected frequency data used for calculating the probabilities.

        truncate: float
            The truncation threshold value. Only values up to this threshold are considered.

    Returns:
        float
            The computed negative log-likelihood. The result is obtained by summing the 
            negative logarithm of the likelihood value.

    Notes:
        - **Version 1**: Ensures compatibility with R/OpenEBGM by calculating the likelihood 
          using `f_star1` and `f_star2`, which represent the adjusted negative binomial 
          probabilities. The likelihood is clipped to avoid log(0) issues.
        
        - **Version 2**: The original Python/Vigipy version calculates the likelihood using 
          `term1` and `term2`, which represent the unadjusted probabilities. This version is 
          commented out in the code but was used in earlier implementations.

        - The function returns the sum of the negative log-likelihood, with a small constant 
          added to avoid taking the logarithm of zero.
    """
    prob1 = p[1] / (p[1] + E)
    prob2 = p[3] / (p[3] + E)
    
    # Vectorized calculation of negative binomial distributions
    dnb1 = np.nan_to_num(nbinom.pmf(n11, p[0], prob1))
    dnb2 = np.nan_to_num(nbinom.pmf(n11, p[2], prob2))
    
    # Vectorized calculation of cumulative distribution functions
    pnb1 = np.nan_to_num(nbinom.cdf(truncate, p[0], prob1))
    pnb2 = np.nan_to_num(nbinom.cdf(truncate, p[2], prob2))
    
    # Check for small denominators
    eps = np.finfo(np.float64).eps
    f_star1 = np.where((1 - pnb1) < eps, 1.0e20, dnb1 / (1 - pnb1))
    f_star2 = np.where((1 - pnb2) < eps, 1.0e20, dnb2 / (1 - pnb2))
    
    L = p[4] * f_star1 + (1 - p[4]) * f_star2
    L = np.clip(L, 1.0e-10, None)
    
    return np.sum(-np.log(L))

    ## This version is the original python/Vigipy version of this 

    # term1 = np.clip(term1, 1e-10, None)
    # term2 = np.clip(term2, 1e-10, None)
    # Return the sum of the negative log-likelihood
    # return np.sum(-np.log(term1 / term2))
    
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def truncated_squashed_likelihood(theta, ni, ei, wi, truncate):
    """
    Calculate the negative log-likelihood for a mixture of two negative binomial distributions, with squashed data.

    Parameters:
    theta (array-like): Parameters of the model. 
                        theta[0] = alpha1 (size parameter for the first distribution)
                        theta[1] = beta1 (parameter for the first distribution)
                        theta[2] = alpha2 (size parameter for the second distribution)
                        theta[3] = beta2 (parameter for the second distribution)
                        theta[4] = P (mixing proportion)
    ni (array-like): Observed counts.
    ei (array-like): Exposure or offset values.
    wi (array-like): Weights for the observations.
    N_star (int, optional): Upper limit for the cumulative distribution function. Default is 1.

    Returns:
    float: The negative log-likelihood value.
    """  
    
    size_f1 = theta[0]  # alpha1
    prob_f1 = theta[1] / (theta[1] + ei)  # beta1 / (beta1 + E)
    size_f2 = theta[2]  # alpha2
    prob_f2 = theta[3] / (theta[3] + ei)  # beta2 / (beta2 + E)
    
    f1 = np.nan_to_num(nbinom.pmf(ni, size_f1, prob_f1))
    f2 = np.nan_to_num(nbinom.pmf(ni, size_f2, prob_f2))
    
    f1_cumul = np.nan_to_num(nbinom.cdf(truncate, size_f1, prob_f1))
    f2_cumul = np.nan_to_num(nbinom.cdf(truncate, size_f2, prob_f2))

    # check whether are are going to run into errors for small denominators
    eps = np.finfo(np.float64).eps
    f_star1 = np.where((1 - f1_cumul) < eps, 1.0e20, f1 / (1 - f1_cumul))
    f_star2 = np.where((1 - f2_cumul) < eps, 1.0e20, f2 / (1 - f2_cumul))
    
    P = theta[4]
    L = (P * f_star1) + ((1 - P) * f_star2)
    L = np.clip(L, 1.0e-10, None)
    
    logL = wi * np.log(L)
    
    return np.sum(-logL)

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def hyperparameter_optimization(
    container,
    truncate,
    truncate_thres,
    priors,
    n11,
    expected,
    minimization_method="CG",
    minimization_bounds=((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1)),
    minimization_options=None,
    message=False,
    opt_likelihood=True,
    number_of_iterations=1000,
    tol_value=1.0e-4,
    sim_anneal=False,
    callback_freq=50
):
    """
    Optimize hyperparameters using likelihood minimization methods.

    This function performs hyperparameter optimization by minimizing a likelihood function, 
    either using non-truncated or truncated data, and can employ various optimization methods 
    including simulated annealing.

    Parameters:
        container: object
            A data container holding the contingency data required for optimization.
        truncate: bool
            If True, apply truncation to the data before optimization.
        truncate_thres: int
            The threshold value used for truncation.
        priors: array-like
            Initial parameter estimates for the optimization process.
        n11: array-like
            Observed frequency data.
        expected: array-like
            Expected frequency data.
        minimization_method: str, optional, default="CG"
            The optimization method to use, such as 'CG' for Conjugate Gradient, or 'DA' for Dual Annealing.
        minimization_bounds: tuple of tuples, optional
            Bounds for the optimization parameters. If the method does not support bounds, 
            this will be ignored. Default bounds provided.
        minimization_options: dict, optional
            Additional options to pass to the optimization function.
        message: bool, optional, default=False
            If True, print messages about the optimization process.
        opt_likelihood: bool, optional, default=True
            If True, use the optimized version of the likelihood function.
        number_of_iterations: int, optional, default=1000
            Maximum number of iterations for the optimization algorithm.
        tol_value: float, optional, default=1.0e-4
            Tolerance value for the optimization convergence criterion.
        sim_anneal: bool, optional, default=False
            If True, use simulated annealing for optimization.
        callback_freq: int, optional, default=50
            How many iterations to call callback after

    Returns:
        p_out: OptimizeResult
            The result of the optimization process, including optimized parameters.

    Notes:
        The function uses `dual_annealing` for simulated annealing if `sim_anneal` is True.
        Otherwise, it uses the specified `minimization_method`. If the method does not 
        support bounds, the bounds parameter is ignored.
    """
    
    # Handle simulated annealing option
    if sim_anneal:
        if message:
            print("USING SIMULATED ANNEALING: EXPERIMENTAL")
        minimization_method = "DA"
    
    # Check if method supports bounds
    if minimization_method not in BOUNDED_METHODS:
        if message:
            print("WARNING: METHOD CHOSEN DOES NOT SUPPORT BOUNDS, THESE ARE NOT BEING APPLIED")
        minimization_bounds = None

    # Set default options for minimization
    if minimization_options is None:
        minimization_options = {}

    # Handle non-truncated case
    if not truncate:
        data_cont = container.contingency
        n1__mat = data_cont.sum(axis=1)
        n_1_mat = data_cont.sum(axis=0)
        rep = len(n_1_mat)
        n1__c = np.tile(n1__mat.values, reps=rep)
        rep = len(n1__mat)
        n_1_c = np.repeat(n_1_mat.values, repeats=rep)
        E_c = np.asarray(n1__c, dtype=np.float64) * n_1_c / N
        n11_c_temp = []
        for col in data_cont:
            n11_c_temp.extend(list(data_cont[col]))
        n11_c = np.asarray(n11_c_temp)

        if opt_likelihood:
            p_out = _optimize_likelihood(
                non_truncated_likelihood_optimised,
                priors, n11_c, E_c, minimization_method, 
                minimization_bounds, minimization_options, 
                number_of_iterations, tol_value, callback_freq
            )
        else:
            p_out = _optimize_likelihood(
                non_truncated_likelihood,
                priors, n11_c, E_c, minimization_method, 
                minimization_bounds, minimization_options, 
                number_of_iterations, tol_value, callback_freq
            )
    
    # Handle truncated case
    else:
        truncate_number = truncate_thres - 1
        n11_truncated = n11[n11 >= truncate_thres]
        expected_truncated = expected[n11 >= truncate_thres]

        callback = Callback(callback_freq)

        likelihood_func = truncated_likelihood_optimised if opt_likelihood else truncated_likelihood
        p_out = minimize(
            likelihood_func,
            x0=priors,
            args=(n11_truncated, expected_truncated, truncate_number),
            options={"maxiter": number_of_iterations},
            tol=tol_value,
            method=minimization_method,
            bounds=minimization_bounds,
            callback=callback,
            **minimization_options
        )

    return p_out

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def hyperparameter_opt_squash(
    truncate,
    truncate_thres,
    priors,
    n_squashed,
    e_squashed,
    weights_squashed,
    minimization_method="CG",
    minimization_bounds=((EPS, 20), (EPS, 10), (EPS, 20), (EPS, 10), (0, 1)),
    minimization_options=None,
    message=False,
    number_of_iterations=1000,
    tol_value=1.0e-4,
    sim_anneal=False,
    callback_freq=50
):
    """
    Optimize hyperparameters using likelihood minimization methods.

    This function performs hyperparameter optimization by minimizing a likelihood function, 
    either using non-truncated or truncated data, and can employ various optimization methods 
    including simulated annealing.

    Parameters:
        truncate: bool
            If True, apply truncation to the data before optimization.
        truncate_thres: int
            The threshold value used for truncation.
        priors: array-like
            Initial parameter estimates for the optimization process.
        n_squashed: array-like
            Observed frequency data (squashed)
        e_squashed: array-like
            Expected frequency data (squashed)
        weights_squashed: array_like
            Weight data (squashed)
        minimization_method: str, optional, default="CG"
            The optimization method to use, such as 'CG' for Conjugate Gradient, or 'DA' for Dual Annealing.
        minimization_bounds: tuple of tuples, optional
            Bounds for the optimization parameters. If the method does not support bounds, 
            this will be ignored. Default bounds provided.
        minimization_options: dict, optional
            Additional options to pass to the optimization function.
        message: bool, optional, default=False
            If True, print messages about the optimization process.
        opt_likelihood: bool, optional, default=True
            If True, use the optimized version of the likelihood function.
        number_of_iterations: int, optional, default=1000
            Maximum number of iterations for the optimization algorithm.
        tol_value: float, optional, default=1.0e-4
            Tolerance value for the optimization convergence criterion.
        sim_anneal: bool, optional, default=False
            If True, use simulated annealing for optimization.
        callback_freq: int, optional, default=50
            After how many iterations to call callback function

    Returns:
        p_out: OptimizeResult
            The result of the optimization process, including optimized parameters.

    Notes:
        The function uses `dual_annealing` for simulated annealing if `sim_anneal` is True.
        Otherwise, it uses the specified `minimization_method`. If the method does not 
        support bounds, the bounds parameter is ignored.
    """
    
    # Handle simulated annealing option
    if sim_anneal:
        raise Exception("Simulated annealing is not implemented for this method yet")
    
    # Check if method supports bounds
    if minimization_method not in BOUNDED_METHODS:
        if message:
            print("WARNING: METHOD CHOSEN DOES NOT SUPPORT BOUNDS, THESE ARE NOT BEING APPLIED")
        minimization_bounds = None

    # Set default options for minimization
    if minimization_options is None:
        minimization_options = {}
        
    truncate_number = truncate_thres - 1
    
    callback = Callback(callback_freq)
    
    p_out = minimize(
        truncated_squashed_likelihood,
        x0=priors,
        args=(n_squashed, e_squashed, weights_squashed, truncate_number),
        options={"maxiter": number_of_iterations},
        tol=tol_value,
        method=minimization_method,
        bounds=minimization_bounds,
        callback=callback,
        **minimization_options
        )

    return p_out

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

# Helper function to reduce code duplication
def _optimize_likelihood(
    likelihood_func, 
    priors, 
    n11_c, 
    E_c, 
    method,
    bounds, 
    options, 
    number_of_iterations, 
    tol_value,
    callback_freq
):
    """
    Optimize a given likelihood function using specified optimization methods.

    This helper function performs optimization of a likelihood function using either 
    dual annealing or other specified optimization methods. It is designed to reduce 
    code duplication and centralize optimization logic.

    Parameters:
        likelihood_func: callable
            The likelihood function to be optimized. It should accept parameters 
            and additional data as inputs and return a scalar value representing 
            the likelihood.
        priors: array-like
            Initial guesses for the parameters of the likelihood function.
        n11_c: array-like
            Observed frequency data used in the likelihood function.
        E_c: array-like
            Expected frequency data used in the likelihood function.
        method: str
            The optimization method to use. It can be 'DA' for dual annealing or 
            other optimization method names supported by `scipy.optimize.minimize`.
        bounds: tuple of tuples, optional
            Bounds for the parameters of the likelihood function. Used only if the 
            chosen method supports parameter bounds.
        options: dict, optional
            Additional options passed to the optimization function.
        message: bool, optional
            If True, print messages about the optimization process.
        number_of_iterations: int, optional, default=1000
            Maximum number of iterations for the optimization algorithm.
        tol_value: float, optional, default=1.0e-4
            Tolerance value for the optimization convergence criterion.
        callback_freq: int, optional, default=50
            After how many iterations to call the callback function

    Returns:
        result: OptimizeResult
            The result of the optimization process, including optimized parameters.

    Notes:
        - If the `method` is 'DA', the function uses `dual_annealing` for optimization.
        - For other methods, it uses `scipy.optimize.minimize` with the specified 
          method and bounds.
        - The `message` parameter is not used directly in this function but may be 
          relevant for debugging or extending the function.
    """
    if method == "DA":
        callback = Anneal_Callback(callback_freq)
        return dual_annealing(
            likelihood_func,
            bounds=bounds,
            args=(n11_c, E_c),
            maxiter=number_of_iterations,
            x0=priors,
            callback=callback
        )
    else:
        callback = Callback(callback_freq)
        return minimize(
            likelihood_func,
            x0=priors,
            args=(n11_c, E_c),
            options={"maxiter": number_of_iterations},
            method=method,
            bounds=bounds,
            tol=tol_value,
            callback=callback,
            **options
        )

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def ebgm_calculation(
    priors,
    expected,
    n11,
    relative_risk
):
    """
    Calculate empirical Bayes estimates for generalized models with truncated data.

    This function computes the empirical Bayes estimates using negative binomial 
    distributions and generalized gamma distributions based on given priors and 
    observed data. It also calculates posterior probabilities and a specific 
    empirical Bayes measure.

    Parameters:
        priors: array-like
            A list or array of prior parameters, where:
            - priors[0]: size parameter for the first negative binomial distribution
            - priors[1]: probability parameter for the first negative binomial distribution
            - priors[2]: size parameter for the second negative binomial distribution
            - priors[3]: probability parameter for the second negative binomial distribution
            - priors[4]: Prior probability weight for the first distribution

        expected: array-like
            Expected frequency values used for calculating the probability parameters 
            of the negative binomial distributions.

        n11: array-like
            Observed frequency data used in the negative binomial probability mass 
            function (PMF) calculations.

        relative_risk: float
            The relative risk value used in the generalized gamma distribution function.

    Returns:
        Qn: array-like
            The weight of the first negative binomial distribution in the posterior 
            probability calculation.

        EBlog2: array-like
            The empirical Bayes estimate of the logarithmic function scaled by the 
            inverse of the logarithm of 2.

        posterior_probability: array-like
            The posterior probability calculated as a weighted sum of generalized gamma 
            distribution probabilities based on the priors and observed data.

    Notes:
        - The function uses the negative binomial PMF from `scipy.stats.nbinom` and 
          a generalized gamma distribution function `gdtr`.
        - The digamma function is used to compute specific logarithmic terms required 
          for the empirical Bayes estimate.
        - The results include the weight of the first distribution, the empirical 
          Bayes estimate, and the posterior probability, all of which are essential 
          for empirical Bayes methods in statistical modeling.

    """
    # Calculate probabilities for nbinom
    prob1 = priors[1] / (priors[1] + expected)
    prob3 = priors[3] / (priors[3] + expected)

    # Negative binomial probabilities
    qdb1 = nbinom.pmf(n11, priors[0], prob1)
    qdb2 = nbinom.pmf(n11, priors[2], prob3)

    # Calculate Qn
    Qn = priors[4] * qdb1 / (priors[4] * qdb1 + (1 - priors[4]) * qdb2)

    # Generalized Gamma Distribution (gdtr equivalent)
    gd1 = gdtr(relative_risk, priors[0] + n11, priors[1] + expected)
    gd2 = gdtr(relative_risk, priors[2] + n11, priors[3] + expected)

    # Posterior probability
    posterior_probability = Qn * gd1 + (1 - Qn) * gd2

    # Digamma and log terms
    dg1 = digamma(priors[0] + n11)
    dgterm1 = dg1 - np.log(priors[1] + expected)
    dg2 = digamma(priors[2] + n11)
    dgterm2 = dg2 - np.log(priors[3] + expected)

    # EBlog2 calculation
    EBlog2 = (np.log(2) ** -1) * (Qn * dgterm1 + (1 - Qn) * dgterm2)

    return Qn, EBlog2, posterior_probability

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def generate_ranking_stats(
    ranking_statistic,
    EBlog2,
    LB,
    posterior_probability,
    num_cell,
    decision_metric,
    decision_thres
):
    """
    Compute ranking statistics and metrics for evaluating model performance.

    This function calculates various statistical measures based on the provided 
    ranking statistic, posterior probabilities, and other metrics. It generates 
    statistics like False Discovery Rate (FDR), False Negative Rate (FNR), 
    Sensitivity (Se), and Specificity (Sp), and determines the number of signals 
    based on the specified decision metric and threshold.

    Parameters:
        ranking_statistic: str
            The type of ranking statistic to compute. Options include:
            - "p_value": Uses posterior probabilities for ranking.
            - "quantile": Uses lower bounds (LB) for ranking.
            - "log2": Uses empirical Bayes estimate of log2 (EBlog2) for ranking.

        EBlog2: array-like
            Empirical Bayes estimates of the logarithmic function scaled by the 
            inverse of the logarithm of 2, used if ranking_statistic is "log2".

        LB: array-like
            Lower bounds used for ranking if ranking_statistic is "quantile".

        posterior_probability: array-like
            Posterior probabilities used for ranking and calculating various metrics.

        num_cell: int
            Total number of cells or observations.

        decision_metric: str
            Metric used to determine the number of signals. Options include:
            - "fdr": Number of signals based on False Discovery Rate.
            - "signals": Number of signals based on direct comparison to the threshold.
            - "rank": Number of signals based on ranking.

        decision_thres: float
            Threshold value used for determining the number of signals based on 
            the chosen decision metric.

    Returns:
        FDR: array-like
            False Discovery Rate computed as the cumulative posterior probability 
            divided by the rank.

        FNR: array-like
            False Negative Rate calculated as the cumulative sum of (1 - posterior 
            probability) divided by the number of remaining cells.

        Se: array-like
            Sensitivity computed as the cumulative sum of posterior probability 
            divided by the total sum of (1 - posterior probability).

        Sp: array-like
            Specificity calculated as the cumulative sum of posterior probability 
            divided by the total number of remaining cells.

        RankStat: array-like
            The ranking statistic used for determining the number of signals.

        num_signals: int
            The number of signals determined based on the decision metric and threshold.

    Notes:
        - The function handles different types of ranking statistics and adjusts 
          calculations accordingly.
        - It includes safeguards against division by zero and ensures proper calculation 
          even with small numbers.
        - The function calculates metrics commonly used in statistical evaluation 
          and decision-making in model performance assessment.
    """

    # Compute RankStat based on the ranking_statistic
    if ranking_statistic == "p_value":
        RankStat = posterior_probability
    elif ranking_statistic == "quantile":
        RankStat = LB
    elif ranking_statistic == "log2":
        RankStat = np.array([x for x in EBlog2])
    

    # Precompute commonly used values
    one_minus_post_prob = 1 - posterior_probability
    post_cumsum = np.cumsum(posterior_probability)
    post_1_cumsum = np.cumsum(one_minus_post_prob)
    post_range = np.arange(1, len(posterior_probability) + 1)

    post_1_sum = post_1_cumsum[-1]  # Sum of (1 - posterior_probability)
    num_minus_post_range = num_cell - post_range + 1e-7  # Prevent division by zero

    # Calculate FDR, FNR, Se, Sp (common calculations for both cases)
    FDR = post_cumsum / post_range
    FNR = np.array(list(reversed(post_1_cumsum))) / num_minus_post_range
    Se = post_1_cumsum / post_1_sum
    Sp = np.array(list(reversed(post_cumsum))) / (num_cell - post_1_sum)

    # Adjust values for "p_value" ranking_statistic case
    if ranking_statistic == "p_value":
        FNR = post_1_cumsum / num_minus_post_range
        Sp = post_cumsum / (num_cell - post_1_sum)

    # Number of signals according to the decision rule (pp/FDR/Nb of Signals)
    if decision_metric == "fdr":
        num_signals = np.sum(FDR <= decision_thres)
    elif decision_metric == "signals":
        num_signals = min(np.sum(RankStat <= decision_thres), num_cell)
    elif decision_metric == "rank":
        if ranking_statistic == "p_value":
            num_signals = np.sum(RankStat <= decision_thres)
        else:  # "quantile" or "log2"
            num_signals = np.sum(RankStat >= decision_thres)

    return FDR, FNR, Se, Sp, RankStat, num_signals

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def local_likelihood_diagnostic(
    container, 
    truncate, 
    truncate_thres, 
    n11, 
    expected, 
    priors,
    N):
    """
    Compute the local likelihood diagnostic based on whether truncation is applied.

    This function calculates the likelihood diagnostic for a given set of priors, 
    either using a truncated or non-truncated likelihood function. The choice of 
    method depends on whether truncation is specified. It processes data accordingly 
    and computes the appropriate likelihood value.

    Parameters:
        container: Data container with contingency data.
            This is used when truncation is not applied. It provides the necessary 
            data for computing the non-truncated likelihood.

        truncate: bool
            Flag indicating whether truncation should be applied in the likelihood 
            calculation.

        truncate_thres: float
            Threshold value used for truncation. Only data points greater than or 
            equal to this threshold are considered when truncation is applied.

        n11: array-like
            Observed frequency data used for likelihood calculations.

        expected: array-like
            Expected frequency data used in conjunction with observed data for likelihood 
            calculations.

        priors: array-like
            Parameters for the likelihood functions. These include prior distributions 
            and other model-specific parameters required for calculating the likelihood.

        N: float
            number of data points

    Returns:
        likeli: float
            The computed likelihood value based on the specified truncation condition. 
            It is calculated using either the `truncated_likelihood_optimised` function 
            if truncation is applied or the `non_truncated_likelihood_optimised` function 
            otherwise.

    Notes:
        - When truncation is applied, the function calculates the likelihood using the 
          truncated version of the likelihood function.
        - When truncation is not applied, the function processes the data from the provided 
          container and calculates the likelihood using the non-truncated function.
        - Ensure that the `container` object and data provided are correctly formatted for 
          the likelihood calculations.
    """
    if (truncate):
        truncate_number = truncate_thres - 1
        n11_truncated = n11[n11 >= truncate_thres]
        expected_truncated = expected[n11 >= truncate_thres]
        likeli = truncated_likelihood_optimised(priors, n11_truncated, expected_truncated, truncate_number)
    else: 
        data_cont = container.contingency
        n1__mat = data_cont.sum(axis=1)
        n_1_mat = data_cont.sum(axis=0)
        rep = len(n_1_mat)
        n1__c = np.tile(n1__mat.values, reps=rep)
        rep = len(n1__mat)
        n_1_c = np.repeat(n_1_mat.values, repeats=rep)
        E_c = np.asarray(n1__c, dtype=np.float64) * n_1_c / N
        n11_c_temp = []
        for col in data_cont:
            n11_c_temp.extend(list(data_cont[col]))
        n11_c = np.asarray(n11_c_temp)
        likeli = non_truncated_likelihood_optimised(priors, n11_c, E_c)
    return likeli

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def ebgm_calculation_inefficient(
    priors,
    expected,
    n11,
    relative_risk
):

    """
    Calculate empirical Bayes estimates for generalized models with truncated data.

    This function computes the empirical Bayes estimates using negative binomial 
    distributions and generalized gamma distributions based on given priors and 
    observed data. It also calculates posterior probabilities and a specific 
    empirical Bayes measure.

    Parameters:
        priors: array-like
            A list or array of prior parameters, where:
            - priors[0]: size parameter for the first negative binomial distribution
            - priors[1]: probability parameter for the first negative binomial distribution
            - priors[2]: size parameter for the second negative binomial distribution
            - priors[3]: probability parameter for the second negative binomial distribution
            - priors[4]: Prior probability weight for the first distribution

        expected: array-like
            Expected frequency values used for calculating the probability parameters 
            of the negative binomial distributions.

        n11: array-like
            Observed frequency data used in the negative binomial probability mass 
            function (PMF) calculations.

        relative_risk: float
            The relative risk value used in the generalized gamma distribution function.

    Returns:
        Qn: array-like
            The weight of the first negative binomial distribution in the posterior 
            probability calculation.

        EBlog2: array-like
            The empirical Bayes estimate of the logarithmic function scaled by the 
            inverse of the logarithm of 2.

        posterior_probability: array-like
            The posterior probability calculated as a weighted sum of generalized gamma 
            distribution probabilities based on the priors and observed data.

    Notes:
        - The function uses the negative binomial PMF from `scipy.stats.nbinom` and 
          a generalized gamma distribution function `gdtr`.
        - The digamma function is used to compute specific logarithmic terms required 
          for the empirical Bayes estimate.
        - The results include the weight of the first distribution, the empirical 
          Bayes estimate, and the posterior probability, all of which are essential 
          for empirical Bayes methods in statistical modeling.

          
    ## This is an old version of the EBGM calculation that is far less efficient that the one we
    ## are currently using in Vigipy

    """


    from sympy.functions.special import gamma_functions
    digamma = np.vectorize(gamma_functions.digamma)

    # old DEPRECATED version
    #qdb1 = dnbinom(n11, size=priors[0], prob=priors[1] / (priors[1] + expected))
    #qdb2 = dnbinom(n11, size=priors[2], prob=priors[3] / (priors[3] + expected))

    prob1 = priors[1]/(priors[1] + expected)
    prob3 = priors[3]/(priors[3] + expected)

    qdb1 = nbinom.pmf(n11, priors[0], prob1)
    qdb2 = nbinom.pmf(n11, priors[2], prob3)

    Qn = priors[4] * qdb1 / (priors[4] * qdb1 + (1 - priors[4]) * qdb2)

    gd1 = gdtr(relative_risk, priors[0] + n11, priors[1] + expected)
    gd2 = gdtr(relative_risk, priors[2] + n11, priors[3] + expected)

    posterior_probability = Qn * gd1 + (1 - Qn) * gd2

    dg1 = digamma(priors[0] + n11)
    dgterm1 = dg1 - np.log(priors[1] + expected)
    dg2 = digamma(priors[2] + n11)
    dgterm2 = dg2 - np.log(priors[3] + expected)

    EBlog2 = (np.log(2) ** -1) * (Qn * dgterm1 + (1 - Qn) * dgterm2)

    return Qn, EBlog2, posterior_probability

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################