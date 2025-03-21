import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Importing Dataset
df = pd.read_csv("heart.csv")

#Reading Train Dataset
print(df.describe())

for column in df.columns:
    plt.figure()  # Create a new figure for each plot
    df[column].hist(bins=10, edgecolor="black")  # Histogram for each column
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(True)
    filename = f"data_visualisation_output\{column}_histogram.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")  # Save with high resolution
    plt.close()  # Close the figure to free memory