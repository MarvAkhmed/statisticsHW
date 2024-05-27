import numpy as np
import random
import matplotlib.pyplot as plt
import statistics
from scipy.stats import kstest, norm
from scipy.stats import f_oneway
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import kruskal
from scipy.stats import chi2

# MARK: - Discrete Random Variable
outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

random_outcome = random.choices(outcomes, probabilities)[0]
print("Random outcome:", random_outcome)

# MARK: -  Continuous Random Variable
random_numbers = np.random.uniform(0, 1, 10)
print("Random numbers from uniform distribution:", random_numbers)

# MARK: - Sampling
# Population of heights (in cm)
population_heights = [170, 175, 180, 185, 190]

# Take a sample of size 3 from the population with replacement
sample_with_replacement = random.choices(population_heights, k=3)
print("Sample with replacement:", sample_with_replacement)

# Take a sample of size 3 from the population without replacement
sample_without_replacement = random.sample(population_heights, k=3)
print("Sample without replacement:", sample_without_replacement)

# Rule: Central Limit Theorem
# Roll a fair six-sided die 1000 times and calculate the mean of each set of rolls
num_rolls = 1000
num_dice = 3  # number of dice rolled each time
means = [np.mean(np.random.randint(1, 7, num_dice)) for _ in range(num_rolls)]

# Plot the distribution of means
plt.hist(means, bins=30, density=True, alpha=0.5)
plt.xlabel('Mean of dice rolls')
plt.ylabel('Probability')
plt.title('Central Limit Theorem: Distribution of Sample Means')
plt.show()

# Rule: Calculating Mean and Median
#  Calculating the mean and median of dataset
data = [10, 15, 20, 25, 30]
mean = statistics.mean(data)
median = statistics.median(data)

print("Mean:", mean)
print("Median:", median)

# Rule: Histogram
# Example: Histogram of patients' blood pressure
blood_pressure_data = [120, 125, 130, 140, 135, 130, 125, 122, 128, 129, 138]
plt.hist(blood_pressure_data, bins=5, color='skyblue', edgecolor='black')
plt.xlabel('Blood Pressure')
plt.ylabel('Frequency')
plt.title('Histogram of Patients\' Blood Pressure')
plt.show()

# Rule: Normality Test
# Generate random blood pressure data (normally distributed)
normal_blood_pressure_data = norm.rvs(loc=120, scale=10, size=100)
# Perform Kolmogorov-Smirnov test
ks_statistic, p_value = kstest(normal_blood_pressure_data, 'norm')

if p_value < 0.05:
    print("Blood pressure data is not normally distributed.")
else:
    print("Blood pressure data is normally distributed.")

# Центральная Предельная Теорема
# Generate 1000 sample means from a uniform distribution
sample_means = [np.mean(np.random.uniform(0, 1, 100)) for _ in range(1000)]

# Plot histogram of sample means
plt.hist(sample_means, bins=30, color='orange', edgecolor='black')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.title('Histogram of Sample Means (Central Limit Theorem)')
plt.show()


# MARK: - LESSON 5 EXERCISE
# Rule: Exercise 1
def exercise_1(scores):
    # Sort the scores
    sorted_scores = sorted(scores)

    # Compute the frequencies of each score
    frequency_dict = {}
    for score in sorted_scores:
        frequency_dict[score] = frequency_dict.get(score, 0) + 1

    # Represent as a variational series
    variational_series = [(value, frequency_dict[value]) for value in sorted(frequency_dict.keys())]

    # Compute sample mean
    sample_mean = sum(scores) / len(scores)

    # Compute sample variance
    sample_variance = sum((x - sample_mean) ** 2 for x in scores) / (len(scores) - 1)

    return variational_series, sample_mean, sample_variance


#  Test the function
scores_1 = [5, 3, 0, 1, 4, 2, 5, 4, 1, 5]
variational_series_1, sample_mean_1, sample_variance_1 = exercise_1(scores_1)
print("Exercise 1:")
print("Variational Series:", variational_series_1)
print("Sample Mean:", sample_mean_1)
print("Sample Variance:", sample_variance_1)


# Rule: Exercise 2
def exercise_2(height_measurements):
    # Define intervals
    intervals = [(150, 155), (155, 160), (160, 165), (165, 170), (170, 175), (175, 180), (180, 185), (185, 190)]

    # Initialize frequencies for each interval
    frequency_dict = {interval: 0 for interval in intervals}

    # Populate frequencies for each interval
    for height in height_measurements:
        for interval in intervals:
            if interval[0] <= height < interval[1]:
                frequency_dict[interval] += 1

    # Represent as an interval variational series
    interval_variational_series = [(interval, frequency_dict[interval]) for interval in intervals]

    # Compute sample mean
    sample_mean = sum(height_measurements) / len(height_measurements)

    # Compute sample variance
    sample_variance = sum((x - sample_mean) ** 2 for x in height_measurements) / (len(height_measurements) - 1)

    return interval_variational_series, sample_mean, sample_variance


# Test the function
height_measurements_2 = [178, 160, 154, 183, 155, 153, 167, 186, 163, 155, 157, 175, 170, 166, 159, 173, 182, 167, 171,
                         169, 179, 165, 156, 179, 158, 171, 175, 173, 164, 172]
interval_variational_series_2, sample_mean_2, sample_variance_2 = exercise_2(height_measurements_2)
print("\nExercise 2:")
print("Interval Variational Series:", interval_variational_series_2)
print("Sample Mean:", sample_mean_2)
print("Sample Variance:", sample_variance_2)


# Rule: Exercise 3
def calculate_mean_and_variance(data):
    # Calculate sample mean
    mean = sum(data) / len(data)

    # Calculate sample variance
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)

    return mean, variance


# Test the function with provided data
data_3 = [178, 160, 154, 183, 155, 153, 167, 186, 163, 155, 157, 175, 170, 166, 159, 173, 182, 167, 171, 169, 179, 165,
          156, 179, 158, 171, 175, 173, 164, 172]
sample_mean_3, sample_variance_3 = calculate_mean_and_variance(data_3)
print("\nExercise 3:")
print("Sample Mean:", sample_mean_3)
print("Sample Variance:", sample_variance_3)

# LESSON 7
# Given data
shelf_heights = {
    'Knee': [80, 75, 78, 85, 79, 82, 77, 81],
    'Waist': [85, 88, 84, 90, 87, 86, 89, 91],
    'Eye': [92, 95, 89, 94, 93, 91, 90, 96]
}

# Perform ANOVA test
f_statistic, p_value = f_oneway(*shelf_heights.values())

# Define significance level
alpha = 0.01

# Make decision
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in mean daily sales among shelf heights.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in mean daily sales among shelf heights.")

# Lesson 8
# Example data for Mann-Whitney U test
group1 = [100, 85, 70, 40, 65, 80, 50, 51, 99, 42]  # Happiness levels before exam
group2 = [80, 75, 90, 50, 40, 20, 50, 40, 92, 73]    # Happiness levels after exam

# Perform Mann-Whitney U test
statistic, p_value = mannwhitneyu(group1, group2)

# Define significance level
alpha = 0.05

# Make decision
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in happiness levels before and after the exam.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in happiness levels before and after the exam.")


# Example data for Wilcoxon Signed-Rank test
before_exam = [100, 85, 70, 40, 65, 80, 50, 51, 99, 42]  # Happiness levels before exam
after_exam = [80, 75, 90, 50, 40, 20, 50, 40, 92, 73]    # Happiness levels after exam

# Perform Wilcoxon Signed-Rank test
statistic, p_value = wilcoxon(before_exam, after_exam)

# Define significance level
alpha = 0.05

# Make decision
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in happiness levels before and after the exam.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in happiness levels before and after the exam.")



# Example data for Kruskal-Wallis test
group0 = [23, 41, 54, 66, 78]  # No anxiety
group1 = [45, 55, 60, 70, 72]  # Low-medium anxiety
group2 = [20, 30, 34, 40, 44]  # High anxiety

# Perform Kruskal-Wallis test
statistic, p_value = kruskal(group0, group1, group2)

# Define significance level
alpha = 0.05

# Make decision
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference in anxiety levels among the groups.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in anxiety levels among the groups.")

# Given data (sample observations)
observed_frequencies = [15, 20, 25, 30, 35]  # Example observed frequencies for different categories/bins
total_obs = sum(observed_frequencies)

# Define the expected frequencies for a specific distribution (e.g., normal distribution)
# You need to specify the expected frequencies based on your null hypothesis
expected_frequencies = [20, 20, 20, 20, 20]  # Example expected frequencies for a uniform distribution

# Calculate the degrees of freedom (df)
df = len(observed_frequencies) - 1

# Calculate the chi-square statistic
chi_square_statistic = sum((observed - expected) ** 2 / expected for observed, expected in zip(observed_frequencies, expected_frequencies))

# Calculate the critical chi-square value
alpha = 0.05  # Significance level
critical_value = chi2.ppf(1 - alpha, df)

# Make decision
if chi_square_statistic > critical_value:
    print("Reject the null hypothesis. The sample data does not suport the null hypothesis.")
else:
    print("Fail to reject the null hypothesis. The sample data supports the null hypothesis.")

# Exercise 2
# Given observed and expected frequencies for different categories
observed_frequencies_2 = [120, 130, 140, 110, 100]
expected_frequencies_2 = [125, 125, 125, 125, 125]

# Calculate the degrees of freedom (df)
df_2 = len(observed_frequencies_2) - 1

# Perform chi-square goodness of fit test
chi_square_statistic_2 = sum((observed - expected) ** 2 / expected for observed, expected in zip(observed_frequencies_2, expected_frequencies_2))

# Calculate the critical chi-square value
alpha_2 = 0.05  # Significance level
critical_value_2 = chi2.ppf(1 - alpha_2, df_2)

# Make decision
if chi_square_statistic_2 > critical_value_2:
    print("Reject the null hypothesis. The sample data does not support the null hypothesis.")
else:
    print("Fail to reject the null hypothesis. The sample data supports the null hypothesis.")
