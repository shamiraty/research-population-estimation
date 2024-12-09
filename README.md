## QUANTITATIVE RESEARCH
## ESTIMATING POPULATION PARAMETERS
## USING SAMPLE ESTIMATORS UNDER CONFIDENCE INTERVALS
## PYTHON PROGRAMMING LANGUAGE

**Note**: Before beginning the analysis, ensure that your data is cleaned by removing any null values and outliers. You can refer to the previous tutorial for guidance on data cleaning: [Data Collection and Cleaning](https://github.com/shamiraty/data-collection-and-cleaning-).

## 1. INTRODUCTION

In research, it is important to estimate population parameters, such as the mean and standard deviation, based on sample data. This helps ensure the results are statistically valid and can be generalized to the entire population. Estimating parameters like the mean allows researchers to understand key trends within the population without needing to collect data from every individual. Confidence intervals (CIs) are used to quantify the uncertainty in these estimates, providing a range of values within which the true population parameter is likely to fall.

Confidence intervals are widely used in various fields, such as health, economics, and social sciences. By applying CIs, researchers can report estimates that are backed by statistical certainty, reducing bias and providing clearer insights for decision-making.

**1.1 Estimation of the Population Mean & Standard Dev in the Study**

This study uses field-collected sample data to estimate the mean age of a population comprising 60,000 individuals. The sample mean serves as the estimator, and the results are presented with a 95% confidence level, ensuring a maximum error probability of 5%.

**1.2 Collected Sample Data**

| Metric                                       | Value                  |
|---------------------------------------------|------------------------|
| Sample Mean (Age)                           | 41.89                  |
| Sample Standard Deviation (Age)             | 14.08                  |
| Population Size (N)                         | 60,000                 |

**1.3 Final Estimated Population Parameters**

The estimated population mean lies within the following confidence interval:

| Metric                                      | Value                  |
|---------------------------------------------|------------------------|
| FPCF Applied                                | No                     |
| Confidence Level                            | 95.0%                  |
| Confidence Interval for Population Mean (Age) | (41.02, 42.76)         |
| Confidence Interval for Population Standard Deviation (Age) | (13.49, 14.73)         |
| Standard Error of the Mean (SEM)            | 0.45                   |

## 2. PROBLEM

In our case, we have a population of 60,000 individuals, but we do not know the exact mean age of this population. To address this, we estimate the mean age using a sample of data, under the assurance of a 95% confidence level and a probability of 0.5 for committing an error. While sampling is crucial for this estimation, errors can occur, and the Standard Error (SE) is used to measure and account for these errors in our estimation.

## 3. IMPORTANCE OF THIS PROJECT

This project demonstrates how to estimate population parameters, specifically the mean and standard deviation, from a sample, using confidence intervals. By applying this methodology, we can provide valid estimates for the entire population, which is essential for making informed decisions based on sample data. Additionally, the application of the Finite Population Correction Factor (FPCF) ensures that our estimates are adjusted for the finite size of the population, improving the accuracy of our confidence intervals.

## 4. METHODOLOGY

In this project, we use the following models and methods:

- **Pandas** for handling and processing data.
- **NumPy** for performing numerical calculations, such as mean and standard deviation.
- **Plotly** for visualizing the normal distribution and confidence intervals.
- **SciPy** for calculating statistical values, such as critical z-values and chi-squared distributions, to construct confidence intervals for the population mean and standard deviation.

**4.1 Code Walkthrough**
**4.2 importing Libraries**

```python
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.stats import chi2, norm
```


**4.3 Load the dataset**
```python
df = pd.read_csv('datasets2.csv')
```

**4.4 Extract the 'age' column**
```python
ages = df['age']
```

**4.5 Sample statistics**
```python
n = len(ages)
sample_mean = np.mean(ages)
sample_std = np.std(ages, ddof=1)  # Use ddof=1 for sample standard deviation
```

**4.6 Population size**
```python
N = 60000
```

**4.7 Confidence level**
```python
confidence_level = 0.95
alpha = 1 - confidence_level
```

**4.8 Check if FPCF should be applied**
```python
fpcf = 1  # Default value (no correction)
if n / N >= 0.05:
    fpcf = np.sqrt((N - n) / (N - 1))
```

**4.9 Confidence interval for population mean**
```python
sem = sample_std / np.sqrt(n)  # Standard error of the mean
sem_corrected = sem * fpcf  # Apply FPCF if necessary
z_critical = norm.ppf(1 - alpha / 2)  # Critical z-value for two-tailed test

margin_of_error_mean = z_critical * sem_corrected
lower_bound_mean = sample_mean - margin_of_error_mean
upper_bound_mean = sample_mean + margin_of_error_mean
```

**4.10 Confidence interval for population standard deviation**
```python
df_degrees = n - 1
chi2_lower = chi2.ppf(alpha / 2, df_degrees)
chi2_upper = chi2.ppf(1 - alpha / 2, df_degrees)

lower_bound_std = np.sqrt((df_degrees * sample_std ** 2) / chi2_upper)
upper_bound_std = np.sqrt((df_degrees * sample_std ** 2) / chi2_lower)
```

**4.11 Define the normal distribution parameters**
```python
x = np.linspace(sample_mean - 4 * sem_corrected, sample_mean + 4 * sem_corrected, 1000)
y = norm.pdf(x, sample_mean, sem_corrected)
```
**4.12 Create figure using Plotly**
```python
fig = go.Figure()

# Add the normal distribution curve
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Distribution'))

# Calculate y values for shading under the curve
shade_x = np.linspace(sample_mean - z_critical * sem_corrected, sample_mean + z_critical * sem_corrected, 100)
shade_y = norm.pdf(shade_x, sample_mean, sem_corrected)

# Add shaded area for the confidence interval
fig.add_trace(go.Scatter(x=np.concatenate([shade_x, shade_x[::-1]]),
                         y=np.concatenate([shade_y, np.zeros_like(shade_x)]),
                         fill='toself', fillcolor='rgba(0,100,80,0.3)',
                         line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval',
                         hoverinfo="skip"))

# Add markers for sample mean and confidence interval boundaries
fig.add_trace(go.Scatter(x=[sample_mean], y=[norm.pdf(sample_mean, sample_mean, sem_corrected)], mode='markers',
                         marker=dict(color='red', size=10), name='Sample Mean'))
fig.add_trace(go.Scatter(x=[lower_bound_mean, upper_bound_mean],
                         y=[norm.pdf(lower_bound_mean, sample_mean, sem_corrected), norm.pdf(upper_bound_mean, sample_mean, sem_corrected)],
                         mode='markers', marker=dict(color='green', size=10), name='CI Bounds'))

# Update layout
fig.update_layout(title='Normal Distribution with Confidence Interval',
                  xaxis_title='Age',
                  yaxis_title='Probability Density',
                  showlegend=True)
```
**4.13 Display key metrics**
```python
print("Sample Mean (Age):", round(sample_mean, 2))
print("Sample Standard Deviation (Age):", round(sample_std, 2))
print("Population Size (N):", N)
print(f"FPCF Applied: {'Yes' if fpcf != 1 else 'No'}")
print(f"Confidence Level: {confidence_level * 100}%")
print("Confidence Interval for Population Mean (Age): (", round(lower_bound_mean, 2), ",", round(upper_bound_mean, 2), ")")
print("Confidence Interval for Population Standard Deviation (Age): (", round(lower_bound_std, 2), ",", round(upper_bound_std, 2), ")")
print("Standard Error of the Mean (SEM):", round(sem, 2))
```

| Statistic                                         | Value               |
|---------------------------------------------------|---------------------|
| Sample Mean (Age)                                 | 41.89               |
| Sample Standard Deviation (Age)                   | 14.08               |
| Population Size (N)                               | 60,000              |
| FPCF Applied                                      | No                  |
| Confidence Level                                  | 95.0%               |
| Confidence Interval for Population Mean (Age)     | (41.02, 42.76)      |
| Confidence Interval for Population Std. Dev. (Age)| (13.49, 14.73)      |
| Standard Error of the Mean (SEM)                  | 0.45                |

**4.14 Display Plotly figure**
```python
fig.show()
```
![9](https://github.com/user-attachments/assets/d18f61af-f83a-40ab-b254-643b50cc1e09)
