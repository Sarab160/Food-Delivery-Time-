import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("Food.csv")  # replace with your CSV file path

# ------------------ BOX PLOT ------------------
plt.figure(figsize=(8,6))
sns.boxplot(x="Traffic_Level", y="Delivery_Time_min", data=df, palette="Set3")
plt.title("Delivery Time by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Delivery Time (minutes)")
plt.savefig("boxplot_delivery_time.png")  # Save image
plt.show()

# ------------------ PAIRPLOT ------------------
# Selecting only numeric columns + categorical for hue
num_cols = ["Distance_km", "Preparation_Time_min", "Delivery_Time_min"]
sns.pairplot(df[num_cols + ["Traffic_Level"]], hue="Traffic_Level", palette="Set2")
plt.savefig("pairplot_delivery.png")  # Save image
plt.show()