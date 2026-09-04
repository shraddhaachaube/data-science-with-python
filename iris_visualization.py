# Week 2 Task - Advanced Data Visualization and Storytelling with Python
# Dataset: Iris Flower Dataset (built into scikit-learn, publicly known dataset
# originally collected by biologist Ronald Fisher in 1936)

# importing the libraries I need
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# setting a simple style so the graphs look neat
sns.set_style("whitegrid")

# loading the iris dataset
iris = load_iris()

# putting it into a pandas dataframe so it is easier to work with
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# renaming columns to simpler names
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# quick look at the data
print(df.head())
print(df.describe())
print(df['species'].value_counts())


# -------------------------------------------------------
# VISUAL 1: Bar Chart - Average petal length per species
# -------------------------------------------------------
avg_petal_length = df.groupby('species')['petal_length'].mean()

plt.figure(figsize=(7, 5))
avg_petal_length.plot(kind='bar', color=['#4C72B0', '#DD8452', '#55A868'])
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('images/1_bar_avg_petal_length.png', dpi=150)
plt.close()


# -------------------------------------------------------
# VISUAL 2: Histogram - Distribution of sepal length
# -------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.hist(df['sepal_length'], bins=15, color='#4C72B0', edgecolor='black')
plt.title('Distribution of Sepal Length (All Species)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Number of Flowers')
plt.tight_layout()
plt.savefig('images/2_histogram_sepal_length.png', dpi=150)
plt.close()


# -------------------------------------------------------
# VISUAL 3: Scatter Plot - Petal length vs Petal width by species
# -------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', s=70)
plt.title('Petal Length vs Petal Width by Species')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.savefig('images/3_scatter_petal_length_width.png', dpi=150)
plt.close()


# -------------------------------------------------------
# VISUAL 4: Box Plot - Sepal width by species (to show spread and outliers)
# -------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='species', y='sepal_width', palette=['#4C72B0', '#DD8452', '#55A868'])
plt.title('Spread of Sepal Width by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')
plt.tight_layout()
plt.savefig('images/4_boxplot_sepal_width.png', dpi=150)
plt.close()


# -------------------------------------------------------
# VISUAL 5: Heatmap - Correlation between the four measurements
# -------------------------------------------------------
plt.figure(figsize=(6, 5))
corr = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Flower Measurements')
plt.tight_layout()
plt.savefig('images/5_heatmap_correlation.png', dpi=150)
plt.close()


# -------------------------------------------------------
# VISUAL 6: Pair Plot - All measurements compared at once, by species
# -------------------------------------------------------
pair = sns.pairplot(df, hue='species', height=1.8)
pair.fig.suptitle('Pairwise Comparison of All Measurements by Species', y=1.02)
pair.savefig('images/6_pairplot_all_features.png', dpi=150)
plt.close()

print("\nAll 6 visualizations were created and saved in the images folder.")
