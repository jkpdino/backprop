import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('training.csv')

# Assuming the first column is the x-axis and the second column is the y-axis
y = data.iloc[:, 0].values

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
ax.plot(y)

# Set the title and axis labels
ax.set_title('Loss over Time')
ax.set_xlabel('Training Pairs')
ax.set_ylabel('Loss')

# Save the plot as a PNG file
plt.savefig('graph2.png', dpi=300, bbox_inches='tight')

# Show the plot (optional)
# plt.show()