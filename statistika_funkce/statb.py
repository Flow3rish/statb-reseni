from numpy import sqrt as numpy_sqrt


def cohen_effect_size_d(X: float, mu_0: float, s: float):
    """
    returns Cohen's d i.e. effect size
    X: sample mean
    mu_0: null hypothesis population mean
    s: sample standard deviation
    """
    return abs(X - mu_0)/s


def chi2_stat_one_sample(s_var: float, sigma_var: float, n: int):
    """
    returns test statistic of Pearson's chi square distribution

    s_var: sample variance
    sigma_var: null hypothesis population variance
    n: sample size
    """
    return (n - 1)*s_var/sigma_var


def t_stat_one_sample(X: float, mu_0: float, s: float, n: int) -> float:
    """
    returns test statistic of student's T distribution

    X: sample mean
    mu_0: null hypothesis population mean
    s: sample standard deviation
    n: sample size
    """
    return (X - mu_0)*numpy_sqrt(n)/s


def z_stat_one_sample(X: float, mu_0: float, sigma: float, n: int) -> float:
    """
    returns test statistic of normal N(0, 1) distribution

    X: sample mean
    mu_0: null hypothesis population mean
    sigma: population standard deviation
    n: sample size
    """
    return (X - mu_0)*numpy_sqrt(n)/sigma


def pi_stat_one_sample(p: float, pi_0: float, n: float):
    """
    returns test statistic of a proportion

    p: prob. of success of a Bernoulli trial
    pi_0: null hyphotesis population proportion
    n: sample size
    """
    if not 0 <= p <= 1:
        raise ValueError("Invalid probability")
    if not 0 <= pi_0 <= 1:
        raise ValueError("Invalid population proportion")
    return (p - pi_0)/numpy_sqrt(pi_0*(1-pi_0)/n)


def waldo_approximation_test(n, p):
    sample_size = n > 30
    variance = n*p*(1-p) > 9
    return(sample_size, variance)
