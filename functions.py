from typing import List

def calc_count(x: List[float]) -> int:
    """Returns the number of data points in a data set"""
    return len(x)

def calc_minimum(x: List[float]) -> int:
    """Returns the lowest data point in a data set"""
    return min(x)

def calc_maximum(x: List[float]) -> int:
    """Returns the highest data point in a data set"""
    return max(x)

def calc_range(x: List[float]) -> float:
    """Returns the difference between the lowest and highest data points in a 
    data set"""
    return calc_maximum(x) - calc_minimum(x)

def calc_mean(x: List[float]) -> float:
    """Returns the sum of the data points divided by the number of data points 
    in a data set"""
    return sum(x) / calc_count(x)

def calc_median(x: List[float]) -> float:
    """Returns the middle data point in a data set"""
    if calc_count(x) % 2 == 1:
        return sorted(x)[calc_count(x) // 2]
    if calc_count(x) % 2 == 0:
        return ((sorted(x)[(calc_count(x) // 2) - 1]) + \
                (sorted(x)[calc_count(x) // 2])) / 2
    
def calc_quantile(x: List[float], y: float) -> float:
    """Returns the percentile value in the data set"""
    return sorted(x)[int(y * calc_count(x))]

def calc_interquantile_range(x: List[float]) -> float:
    """Returns the difference between the 75th percentile and the 25th 
    percentile in a data set"""
    return calc_quantile(x, 0.75) - calc_quantile(x, 0.25)

def calc_mode(x: List[float]) -> List[float]:
    """Returns the number(s) that appear most frequently in a data set"""
    frequency_dict = {}
    for i in x:
        if i not in frequency_dict:
            frequency_dict[i] = 1
        else:
            frequency_dict[i] += 1
    modes = []
    highest_frequency = sorted(frequency_dict.values())[-1]
    for key, value in frequency_dict.items():
        if value == highest_frequency:
            modes.append(key)
    return modes

def calc_variance(x: List[float]) -> float:
    """Returns the average of the squared differences from the mean in a data 
    set"""
    assert calc_count(x) >= 2, "Variance requires at least two elements"
    return sum([(i - (calc_mean(x)))**2 for i in x]) / calc_count(x)

def calc_bessel_variance(x: List[float]) -> float:
    """Returns the average of the squared differences from the mean in a data 
    set but corrected for bias in the estimation of the population variance"""
    assert calc_count(x) >= 2, "Variance requires at least two elements"
    return sum([(i - (calc_mean(x)))**2 for i in x]) / (calc_count(x) - 1)

def calc_standard_deviation(x: List[float]) -> float:
    """Returns the dispersion of a dataset relative to its mean and is 
    calculated as the square root of the
    variance"""
    assert calc_count(x) >= 2, "Standard Deviation requires at least two " \
    "elements"
    return (sum([(i - (calc_mean(x)))**2 for i in x]) / calc_count(x))**0.5

def calc_bessel_standard_deviation(x: List[float]) -> float:
    """Returns the dispersion of a dataset relative to its mean and is 
    calculated as the square root of the variance but corrected for bias in the 
    estimation of the population"""
    assert calc_count(x) >= 2, "Standard Deviation requires at least two " \
    "elements"
    return (sum([(i - (calc_mean(x)))**2 for i in x]) / \
            (calc_count(x) - 1))**0.5

def calc_covariance(x: List[float], y: List[float]) -> float:
    """Returns a measurement of how changes in one variable are associated with 
    changes in a second variable"""
    assert calc_count(x) == calc_count(y), "x and y must contain the same " \
    "number of elements"
    return sum([(i - (calc_mean(x))) * (j - (calc_mean(y))) for i, \
                j in zip(x, y)]) / (calc_count(x) - 1)

def calc_correlation(x: List[float], y: List[float]) -> float:
    """Returns a measurement of the strength of the relationship between the 
    relative movements of two data sets"""
    if calc_bessel_standard_deviation(x) > 0 and \
    calc_bessel_standard_deviation(y) > 0:
        return calc_covariance(x, y) / calc_bessel_standard_deviation(x) / \
        calc_bessel_standard_deviation(y)
    else:
        return 0