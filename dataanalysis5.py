

import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


print("=" * 50)
print("1. MATH MODULE")
print("=" * 50)

number = 25

print("Square root of", number, ":", math.sqrt(number))
print("Factorial of 5:", math.factorial(5))
print("Ceiling of 4.3:", math.ceil(4.3))
print("Floor of 4.8:", math.floor(4.8))
print("Absolute value of -15:", math.fabs(-15))



print("\n" + "=" * 50)
print("2. NUMPY")
print("=" * 50)

marks = np.array([75, 80, 85, 90, 95])

print("NumPy Array:", marks)
print("Mean:", np.mean(marks))
print("Median:", np.median(marks))
print("Standard Deviation:", np.std(marks))
print("Square Root:", np.sqrt(marks))
print("Array + 5:", marks + 5)


print("\n" + "=" * 50)
print("3. PANDAS DATA ANALYSIS")
print("=" * 50)


data = {
    "Student": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "Maths": [85, 78, 92, 67, 88, 76, 95, 82],
    "Science": [90, 80, 89, 70, 92, 75, 96, 85],
    "English": [78, 85, 88, 72, 90, 80, 91, 84]
}

df = pd.DataFrame(data)

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())



print("\n" + "=" * 50)
print("4. DATA PREPROCESSING")
print("=" * 50)

print("Duplicate rows:", df.duplicated().sum())

df[["Maths", "Science", "English"]] = (
    df[["Maths", "Science", "English"]]
    .fillna(df[["Maths", "Science", "English"]].mean())
)

df["Average"] = df[["Maths", "Science", "English"]].mean(axis=1)

print("\nProcessed dataset:")
print(df)

print("\nAverage marks by student:")
print(df[["Student", "Average"]])


print("\n" + "=" * 50)
print("5. NUMPY STATISTICS")
print("=" * 50)

all_marks = df[["Maths", "Science", "English"]].to_numpy()

print("Overall Mean:", np.mean(all_marks))
print("Overall Median:", np.median(all_marks))
print("Overall Standard Deviation:", np.std(all_marks))


print("\n" + "=" * 50)
print("6. SCIPY STATISTICS")
print("=" * 50)

maths_marks = df["Maths"].to_numpy()
science_marks = df["Science"].to_numpy()

print("Mode of Maths marks:", stats.mode(maths_marks, keepdims=False).mode)
print("Correlation between Maths and Science:",
      stats.pearsonr(maths_marks, science_marks).statistic)




plt.figure(figsize=(8, 5))
plt.bar(df["Student"], df["Average"])
plt.title("Average Marks of Students")
plt.xlabel("Student")
plt.ylabel("Average Marks")
plt.tight_layout()
plt.savefig("average_marks.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(df["Student"], df["Maths"], marker="o", label="Maths")
plt.plot(df["Student"], df["Science"], marker="s", label="Science")
plt.plot(df["Student"], df["English"], marker="^", label="English")
plt.title("Subject-wise Marks")
plt.xlabel("Student")
plt.ylabel("Marks")
plt.legend()
plt.tight_layout()
plt.savefig("subject_marks.png")
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(
    df[["Maths", "Science", "English"]].corr(),
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Between Subjects")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

print("\nAll analysis and visualizations completed successfully.")
