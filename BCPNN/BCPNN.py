import numpy
import pandas as pd
from scipy.stats import norm
from sympy.functions.special import gamma_functions
from ..utils.Container import Container

digamma = numpy.vectorize(gamma_functions.digamma)
trigamma = numpy.vectorize(gamma_functions.trigamma)


def _compute_expected_counts(N, n1j, ni1):
    return n1j * ni1 / N


def bcpnn(container, relative_risk=1, min_events=1, decision_metric='fdr',
          decision_thres=0.05, ranking_statistic='p_value', MC=False,
          num_MC=10000):
    '''
    A Bayesian Confidence Propogation Neural Network. Note on variable
    naming conventions:
    nXX - XX refers to the (i,j) indices of a 2x2 contingency table
          combining product and adverse events as the counts.
          Products are used as rows and AEs as columns.
            For example:
                n11 = events related to a product/AE combo
                n1j = All events related to that particular product

    Arguments:
        container: A DataContainer object produced by the convert()
                    function from data_prep.py

        relative_risk (int/float): The relative risk value

        min_events: The min number of AE reports to be considered a signal

        decision_metric (str): The metric used for detecting signals:
                            {fdr = false detection rate,
                            signals = number of signals,
                            rank = ranking statistic}

        decision_thres (float): The min thres value for the decision_metric

        ranking_statistic (str): How to rank signals:
                            {'p_value' = posterior prob of the null hypothesis,
                            'quantile' = 2.5% quantile of the IC}

        MC (Bool): Use Monte Carlo simulations to make results more robust?

        num_mc (int): Number of MC simulations to run

    '''
    DATA = container.data
    N = container.N

    if min_events > 1:
        DATA = DATA.loc[DATA.events >= min_events]

    n11 = numpy.asarray(DATA['events'], dtype=numpy.float64)
    n1j = numpy.asarray(DATA['product_aes'], dtype=numpy.float64)
    ni1 = numpy.asarray(DATA['count_across_brands'], dtype=numpy.float64)
    E = _compute_expected_counts(N, n1j, ni1)

    n10 = n1j - n11
    n01 = ni1 - n11
    n00 = N - (n11 + n10 + n01)
    num_cell = len(n11)

    if not MC:
        p1 = 1 + n1j
        p2 = 1 + N - n1j
        q1 = 1 + ni1
        q2 = 1 + N - ni1
        r1 = 1 + n11
        r2b = N - n11 - 1 + (2+N)**2 / (q1*p1)
        # Calculate the Information Criterion
        digamma_term = (digamma(r1) - digamma(r1+r2b) -
                        (digamma(p1) - digamma(p1+p2)
                         + digamma(q1) - digamma(q1+q2)))
        IC = numpy.asarray((numpy.log(2)**-1) * digamma_term,
                           dtype=numpy.float64)
        IC_variance = numpy.asarray((numpy.log(2)**-2) *
                                    (trigamma(r1) - trigamma(r1+r2b) +
                                    (trigamma(p1) - trigamma(p1+p2) +
                                     trigamma(q1) - trigamma(q1+q2))),
                                    dtype=numpy.float64)
        posterior_prob = norm.cdf(numpy.log(relative_risk),
                                  IC, numpy.sqrt(IC_variance))
        lower_bound = norm.ppf(0.025, IC, numpy.sqrt(IC_variance))
    else:
        num_MC = float(num_MC)
        # Priors for the contingency table
        q1j = (n1j + .5)/(N + 1)
        qi1 = (ni1 + .5)/(N + 1)
        qi0 = (N - ni1 + .5)/(N + 1)
        q0j = (N - n1j + .5)/(N + 1)

        a_ = .5/(q1j * qi1)

        a11 = q1j * qi1 * a_
        a10 = q1j * qi0 * a_
        a01 = q0j * qi1 * a_
        a00 = q0j * qi0 * a_

        g11 = a11 + n11
        g10 = a10 + n10
        g01 = a01 + n01
        g00 = a00 + n00

        posterior_prob = []
        lower_bound = []
        for m in range(num_cell):
            alpha = [g11[m], g10[m], g01[m], g00[m]]
            p = numpy.random.dirichlet(alpha, int(num_MC))
            p11 = p[:, 0]
            p1_ = p11 + p[:, 1]
            p_1 = p11 + p[:, 2]
            ic_monte = numpy.log(p11 / (p1_ * p_1))
            temp = 1*(ic_monte < numpy.log(relative_risk))
            posterior_prob.append(sum(temp)/num_MC)
            lower_bound.append(ic_monte[round(num_MC * 0.025)])
        posterior_prob = numpy.asarray(posterior_prob)
        lower_bound = numpy.asarray(lower_bound)

    if ranking_statistic == 'p_value':
        RankStat = posterior_prob
    else:
        RankStat = lower_bound

    if ranking_statistic == 'p_value':
        FDR = ((numpy.cumsum(posterior_prob) /
                numpy.arange(1, len(posterior_prob)+1)))
        FNR = ((numpy.cumsum(1-posterior_prob)[::-1]) /
               (num_cell-numpy.arange(1, len(posterior_prob)+1)))
        Se = numpy.cumsum(1-posterior_prob) / (sum(1-posterior_prob))
        Sp = ((numpy.cumsum(posterior_prob)[::-1]) /
              (num_cell - sum(1-posterior_prob)))
    else:
        FDR = ((numpy.cumsum(posterior_prob) /
                numpy.arange(1, len(posterior_prob)+1)))
        FNR = ((numpy.cumsum(1-posterior_prob)[::-1]) /
               (num_cell - numpy.arange(1, len(posterior_prob)+1)))
        Se = numpy.cumsum((1-posterior_prob)) / (sum(1-posterior_prob))
        Sp = ((numpy.cumsum(posterior_prob)[::-1]) /
              (num_cell - sum(1-posterior_prob)))

    if decision_metric == 'fdr':
        num_signals = (FDR <= decision_thres).sum()
        sorter = 'FDR'
    elif decision_metric == 'signals':
        num_signals = min(decision_thres, num_cell)
    elif decision_metric == 'rank':
        if ranking_statistic == 'p_value':
            num_signals = (RankStat <= decision_thres).sum()
            sorter = 'posterior_prob'
        elif ranking_statistic == 'quantile':
            num_signals = (RankStat >= decision_thres).sum()
            sorter = 'Q_0.025(log(IC))'

    name = DATA['product_name']
    ae = DATA['ae_name']
    count = n11
    RC = Container()

    RC.input_param = (relative_risk, min_events, decision_metric,
                      decision_thres, ranking_statistic)

    # SIGNALS RESULTS and presentation
    if ranking_statistic == 'p_value':
        RC.all_signals = pd.DataFrame({'Product': name,
                                       'Adverse Event': ae,
                                       'Count': count,
                                       'Expected Count': E,
                                       'posterior_prob': RankStat,
                                       'count/expected': (count/E),
                                       'product margin': n1j,
                                       'event margin': ni1,
                                       'FDR': FDR,
                                       'FNR': FNR,
                                       'Se': Se,
                                       'Sp': Sp}).sort_values(by=[sorter])
    else:
        RC.all_signals = pd.DataFrame({'Product': name,
                                       'Adverse Event': ae,
                                       'Count': count,
                                       'Expected Count': E,
                                       'Q_0.025(log(IC))': RankStat,
                                       'count/expected': (count/E),
                                       'product margin': n1j,
                                       'event margin': ni1,
                                       'FDR': FDR,
                                       'FNR': FNR,
                                       'Se': Se,
                                       'Sp': Sp}).sort_values(by=[sorter],
                                                              ascending=False)

    # List of Signals generated according to the decision_thres
    RC.all_signals.index = numpy.arange(0, len(RC.all_signals.index))
    if num_signals > 0:
        num_signals -= 1
    else:
        num_signals = 0
    RC.signals = RC.all_signals.loc[0:num_signals, ]

    # Number of signals
    RC.num_signals = num_signals
    return RC