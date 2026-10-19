import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("Heart_attack_risk_prediction.csv");
 
print(df)

print("Handling missing values");
print(df.isnull().sum()) 



df["BMI_Values"] = df["BMI_Values"].fillna(df["BMI_Values"].median())
df["Smoking_Habits"] = df["Smoking_Habits"].fillna("Unknown")
print(df.isnull().sum())

print("Duplicate rows: ",df.duplicated().sum())



Define your target columns
pollutants = ["Glucose_Levels", "BMI_Values"]  # Replace with your column names

for column in pollutants:
    if column in df.columns:
        # Ensure column is numeric and drop NaNs for outlier calculations
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        # Calculate IQR boundaries
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Filter outliers
        outliers = df[(df[column] < lower) | (df[column] > upper)]
        print(f"{column} Outliers: {len(outliers)}")

        # Plot boxplot
        plt.figure(figsize=(6, 4))
        plt.boxplot(series)
        plt.title(f"Box Plot - {column}")
        plt.ylabel(column)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
    else:
        print(f"Column '{column}' not found in DataFrame.")

print(df.describe())

print(question 1")
counts = df["Heart_Attack"].value_counts()
percent = df["Heart_Attack"].value_counts(normalize=True).mul(100).round(2)
print(pd.DataFrame({"Patients": counts, "Percent": percent}))
print()
print("Imbalance ratio (No : Yes) = %.1f : 1" % (counts[0] / counts[1]))
 
print(question 2")
df["Age_Group"] = pd.cut(df["Age"], bins=[0, 20, 40, 60, 90], labels=["0-20", "21-40", "41-60", "61+"])
age_tbl = df.groupby("Age_Group", observed=True)["Heart_Attack"].agg(
    Patients="count", Heart_Attacks="sum", Rate_Percent=lambda s: round(s.mean() * 100, 2))
print(age_tbl)
print()
print("Mean age (no heart attack):", round(df.loc[df.Heart_Attack == 0, "Age"].mean(), 1))
print("Mean age (heart attack)   :", round(df.loc[df.Heart_Attack == 1, "Age"].mean(), 1))

print(question 3")
for col in ["HyperTension", "Heart_Disease"]:
    t = df.groupby(col)["Heart_Attack"].agg(
        Patients="count", Heart_Attacks="sum", Rate_Percent=lambda s: round(s.mean() * 100, 2))
    print(t, "\n")
 
both = df.groupby(["HyperTension", "Heart_Disease"])["Heart_Attack"].mean().mul(100).round(2)
print("Heart attack rate (%) by both conditions:")
print(both)

print(question 4")
print(df.groupby("Heart_Attack")[["Glucose_Levels", "BMI_Values"]].agg(["mean", "median"]).round(2))
print()
df["Glucose_Group"] = pd.cut(df["Glucose_Levels"], bins=[0, 100, 126, 200, 300],
                             labels=["Normal (<100)", "Pre-diabetic (100-125)",
                                     "High (126-199)", "Very high (200+)"], right=False)
print(df.groupby("Glucose_Group", observed=True)["Heart_Attack"].agg(
    Patients="count", Rate_Percent=lambda s: round(s.mean() * 100, 2)))

print(question 5")
for col in ["Smoking_Habits", "Is_Married", "Employment_Type", "Gender", "Residential_type"]:
    t = df.groupby(col, dropna=False)["Heart_Attack"].agg(
        Patients="count", Rate_Percent=lambda s: round(s.mean() * 100, 2))
    print(t, "\n")
 
print("BAR_CHART")
Create age groups
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[-1, 20, 40, 60, 100],
    labels=["0-20", "21-40", "41-60", "61+"]
)

# Calculate heart attack percentage
rates = df.groupby("Age_Group", observed=False)["Heart_Attack"].mean() * 100

# Create bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(rates.index, rates.values, color="#c0392b")

# Display percentages above bars
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.1,
        f"{bar.get_height():.2f}%",
        ha="center"
    )

plt.title("Heart Attack Rate by Age Group")
plt.xlabel("Age group (years)")
plt.ylabel("Heart attack rate (%)")
plt.grid(True, alpha=0.5)
plt.show()

print(" SCATTER PLOT")
plt.figure(figsize=(10, 5))

sns.scatterplot(
    data=df,
    x="Age",
    y="Glucose_Levels",
    hue="Heart_Attack",
    palette={0: "steelblue", 1: "#c0392b"},
    alpha=0.6,
    s=25
)

plt.title("Age vs Glucose Level by Heart Attack Outcome")
plt.xlabel("Age (years)")
plt.ylabel("Glucose level")

plt.legend(
    title="Outcome",
    labels=["No heart attack", "Heart attack"]
)

plt.grid(True, alpha=0.5)
plt.show()


print(" HEATMAP")
# Select numerical columns
columns = [
    "Age",
    "HyperTension",
    "Heart_Disease",
    "Glucose_Levels",
    "BMI_Values",
    "Heart_Attack"
]

# Calculate correlations
correlation = df[columns].corr()

# Create heatmap
plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="Reds",
    linewidths=0.5,
    vmin=0,
    vmax=1
)

plt.title("Correlation Heatmap of Numeric Variables")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()'''
'''
print("HISTOGRAM")
# Divide dataset based on outcome
no_attack = df[df["Heart_Attack"] == 0]
attack = df[df["Heart_Attack"] == 1]

# Create histogram
plt.figure(figsize=(9, 5))

plt.hist(
    no_attack["Age"],
    bins=30,
    density=True,
    alpha=0.4,
    color="steelblue",
    label="No heart attack"
)

plt.hist(
    attack["Age"],
    bins=30,
    density=True,
    alpha=0.4,
    color="#c0392b",
    label="Heart attack"
)

plt.title("Age Distribution: Heart Attack vs No Heart Attack")
plt.xlabel("Age (years)")
plt.ylabel("Density")
plt.legend(title="Outcome")
plt.grid(True, alpha=0.5)
plt.show()


print("BOX PLOT")
# Create outcome labels
df["Outcome"] = df["Heart_Attack"].map({
    0: "No heart attack",
    1: "Heart attack"
})

# Create two box plots
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Colors for outcomes
colors = ["steelblue", "#c0392b"]

# Glucose level box plot
sns.boxplot(
    data=df,
    x="Outcome",
    y="Glucose_Levels",
    hue="Outcome",
    palette=colors,
    legend=False,
    ax=axes[0]
)

axes[0].set_title("Glucose Level by Outcome")
axes[0].set_xlabel("")
axes[0].set_ylabel("Glucose Levels")

# BMI box plot
sns.boxplot(
    data=df,
    x="Outcome",
    y="BMI_Values",
    hue="Outcome",
    palette=colors,
    legend=False,
    ax=axes[1]
)

axes[1].set_title("BMI by Outcome")
axes[1].set_xlabel("")
axes[1].set_ylabel("BMI Values")


colors = {
    "No heart attack": "steelblue",
    "Heart attack": "#c0392b"
}

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

sns.boxplot(
    data=df,
    x="Outcome",
    y="Glucose_Levels",
    hue="Outcome",
    palette=colors,
    legend=False,
    ax=axes[0]
)
axes[0].set_title("Glucose Level by Outcome")
axes[0].set_xlabel("")

sns.boxplot(
    data=df,
    x="Outcome",
    y="BMI_Values",
    hue="Outcome",
    palette=colors,
    legend=False,
    ax=axes[1]
)
axes[1].set_title("BMI by Outcome")
axes[1].set_xlabel("")

plt.tight_layout()
plt.show()