# Week 3 Task - Statistical Analysis and Hypothesis Testing
# Topic: Does employee training improve performance scores?

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Create the dataset
# -----------------------------
# I did not find a ready-made dataset for this exact question,
# so I generated a sample dataset myself (self-generated data)
# to represent employee performance scores (out of 100).

np.random.seed(42)  # so the results stay the same every time I run it

# Group 1: employees who completed the new training program
trained_scores = np.random.normal(loc=78, scale=8, size=40)

# Group 2: employees who did NOT complete the training program
untrained_scores = np.random.normal(loc=72, scale=9, size=40)

# put everything into one dataframe (easier to save/view as a table)
df = pd.DataFrame({
    "score": np.concatenate([trained_scores, untrained_scores]),
    "group": ["Trained"] * 40 + ["Untrained"] * 40
})

df.to_csv("employee_scores.csv", index=False)
print("Dataset saved as employee_scores.csv")
print(df.head())

# -----------------------------
# Step 2: Basic descriptive statistics
# -----------------------------
print("\n--- Descriptive Statistics ---")
print(df.groupby("group")["score"].describe())

trained_mean = trained_scores.mean()
untrained_mean = untrained_scores.mean()
print(f"\nMean score - Trained group: {trained_mean:.2f}")
print(f"Mean score - Untrained group: {untrained_mean:.2f}")

# -----------------------------
# Step 3: Hypothesis
# -----------------------------
# Null Hypothesis (H0): There is no difference in the average performance
#                        score between trained and untrained employees.
# Alternative Hypothesis (H1): Employees who completed the training have a
#                        higher average performance score than employees
#                        who did not.
# Significance level (alpha) = 0.05

alpha = 0.05

# -----------------------------
# Step 4: Independent samples t-test
# -----------------------------
# I used an independent (two-sample) t-test because I am comparing the
# means of two separate, independent groups.

t_stat, p_value = stats.ttest_ind(trained_scores, untrained_scores, equal_var=False)

print("\n--- Independent T-Test Results ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value (two-tailed): {p_value:.4f}")

# Since I am testing if trained > untrained (a one-sided hypothesis),
# I divide the two-tailed p-value by 2 (only valid because t_stat is positive,
# meaning the difference is in the direction we expected)
if t_stat > 0:
    p_value_one_sided = p_value / 2
else:
    p_value_one_sided = 1 - (p_value / 2)

print(f"P-value (one-tailed): {p_value_one_sided:.4f}")

if p_value_one_sided < alpha:
    print("Result: Reject the Null Hypothesis (H0).")
    print("The training program appears to significantly improve scores.")
else:
    print("Result: Fail to reject the Null Hypothesis (H0).")
    print("Not enough evidence that training improves scores.")

# -----------------------------
# Step 5: 95% Confidence Interval for the difference in means
# -----------------------------
mean_diff = trained_mean - untrained_mean
se_diff = np.sqrt(trained_scores.var(ddof=1)/len(trained_scores) +
                   untrained_scores.var(ddof=1)/len(untrained_scores))

# degrees of freedom (Welch's approximation) - using scipy for the t critical value
df_welch = ((trained_scores.var(ddof=1)/len(trained_scores) +
             untrained_scores.var(ddof=1)/len(untrained_scores))**2) / (
    ((trained_scores.var(ddof=1)/len(trained_scores))**2)/(len(trained_scores)-1) +
    ((untrained_scores.var(ddof=1)/len(untrained_scores))**2)/(len(untrained_scores)-1)
)

t_critical = stats.t.ppf(0.975, df_welch)
ci_lower = mean_diff - t_critical * se_diff
ci_upper = mean_diff + t_critical * se_diff

print("\n--- 95% Confidence Interval for Difference in Means ---")
print(f"Mean difference (Trained - Untrained): {mean_diff:.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

# -----------------------------
# Step 6: Visualizations
# -----------------------------

# Chart 1: Boxplot comparing both groups
plt.figure(figsize=(7, 5))
df.boxplot(column="score", by="group", grid=False)
plt.title("Performance Score by Training Group")
plt.suptitle("")  # remove the automatic subtitle pandas adds
plt.xlabel("Group")
plt.ylabel("Performance Score")
plt.savefig("boxplot_scores.png", dpi=150, bbox_inches="tight")
plt.close()

# Chart 2: Histogram comparing both groups
plt.figure(figsize=(7, 5))
plt.hist(trained_scores, bins=10, alpha=0.6, label="Trained")
plt.hist(untrained_scores, bins=10, alpha=0.6, label="Untrained")
plt.title("Distribution of Performance Scores")
plt.xlabel("Performance Score")
plt.ylabel("Number of Employees")
plt.legend()
plt.savefig("histogram_scores.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nCharts saved: boxplot_scores.png and histogram_scores.png")
print("\nDone!")
