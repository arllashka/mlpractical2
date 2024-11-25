import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('VGG_38_BN_experiment/result_outputs/summary.csv')

# Generate epoch numbers
epochs = range(1, len(data) + 1)

# Create the accuracy plot
plt.figure(figsize=(6, 4))  # Adjust figure size for better aspect ratio
plt.plot(epochs, data['train_acc'], label='Training Accuracy', linestyle='-', color='green')
plt.plot(epochs, data['val_acc'], label='Validation Accuracy', linestyle='-', color='magenta')

# Set plot titles and labels
# plt.title('Classification Accuracy per Epoch', fontsize=10)
plt.xlabel('Epoch Number', fontsize=9)
plt.ylabel('Accuracy', fontsize=9)
plt.legend(fontsize=8)
plt.grid(True)

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the plot as a PDF file
plt.savefig('VGG38_BN_Res_accuracy.pdf', format='pdf', bbox_inches='tight')

# Optionally, display the plot
# plt.show()

