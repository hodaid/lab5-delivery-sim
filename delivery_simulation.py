import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
normal_delivery = np.random.normal(loc=31.5, scale=6.5, size=97)
long_outliers = np.array([68.0, 71.0, 75.0]) # Outliers taking unusually long
delivery_times = np.concatenate([normal_delivery, long_outliers])

data = {
    "OrderID": range(1, 101),
    "DeliveryTime(min)": delivery_times
}
pd.DataFrame(data).to_csv("delivery_times.csv", index=False)

df = pd.read_csv("delivery_times.csv")

real_mean = df["DeliveryTime(min)"].mean()
real_std = df["DeliveryTime(min)"].std()
print(f"Real delivery times -- Mean: {real_mean:.1f} min Std: {real_std:.1f} min")

simulated_times = np.random.normal(loc=real_mean, scale=real_std, size=100)
print(f"Simulated 100 values from Normal(mean={real_mean:.1f}, std={real_std:.1f})")

plt.figure(figsize=(8, 5))
plt.hist(df["DeliveryTime(min)"], bins=15, alpha=0.6, label="Real Delivery Times", color="steelblue", edgecolor="black")
plt.hist(simulated_times, bins=15, alpha=0.6, label="Simulated Normal Times", color="sandybrown", edgecolor="black")
plt.title("Real vs. Simulated Delivery Times")
plt.xlabel("Delivery Time (min)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("delivery_comparison_histogram.png")
plt.close()

missing_vals = df["DeliveryTime(min)"].isnull().sum()
print(f"Missing values: {missing_vals}")

q25 = df["DeliveryTime(min)"].quantile(0.25)
q50 = df["DeliveryTime(min)"].quantile(0.50)
q75 = df["DeliveryTime(min)"].quantile(0.75)
print(f"Percentiles -- 25th: {q25:.1f} 50th: {q50:.1f} 75th: {q75:.1f}")

plt.figure(figsize=(5, 4))
plt.boxplot(df["DeliveryTime(min)"], patch_artist=True, boxprops=dict(facecolor="indianred"))
plt.title("Real Delivery Times Boxplot")
plt.ylabel("Delivery Time (min)")
plt.tight_layout()
plt.savefig("delivery_boxplot.png")
plt.close()

iqr = q75 - q25
lower_bound = q25 - 1.5 * iqr
upper_bound = q75 + 1.5 * iqr
print(f"IQR outlier bounds: {lower_bound:.1f} - {upper_bound:.1f}")

outliers = df[df["DeliveryTime(min)"] > upper_bound]["DeliveryTime(min)"].tolist()
outliers_int = [int(round(val)) for val in outliers]

print(f"Outliers detected: {len(outliers)} deliveries ({outliers_int} minutes)")

percentage_outliers = (len(outliers) / len(df)) * 100
print(f"Percentage flagged as outliers: {percentage_outliers:.1f}%")

print("Saved delivery_comparison_histogram.png, delivery_boxplot.png")