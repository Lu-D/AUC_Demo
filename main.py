import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# Function to generate a circular ground truth
def generate_circular_ground_truth(shape, center, radius):
    y, x = np.ogrid[:shape[0], :shape[1]]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
    return mask.astype(int)

# Function to calculate ROC AUC score and plot ROC curve
def calculate_roc_auc(gt, preds, axes):
    fpr, tpr, thresholds = metrics.roc_curve(gt.ravel(), preds.ravel())
    roc_auc = metrics.auc(fpr, tpr)

    # Display ground truth
    axes[0].imshow(ground_truth, cmap='rainbow', interpolation='nearest')
    axes[0].set_title('Ground Truth')

    # Display simulated predictions
    axes[1].imshow(simulated_predictions, cmap='rainbow', interpolation='nearest')
    axes[1].set_title('Simulated Predictions')

    # Display AUC-ROC curve
    axes[2].cla()  # Clear the current axis
    axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title('Receiver Operating Characteristic (ROC) Curve')
    axes[2].legend(loc='lower right')

    plt.draw()

# Function to handle mouse click events
# Function to handle mouse click events
def onclick(event):
    global simulated_predictions  # Declare simulated_predictions as a global variable
    if event.inaxes is not None:
        # print('click')
        x = int(event.xdata)
        y = int(event.ydata)

        # Change simulated predictions in a circular region centered around the clicked point
        radius = 20  # You can adjust the radius as needed
        y_circle, x_circle = np.ogrid[:simulated_predictions.shape[0], :simulated_predictions.shape[1]]
        mask = (x_circle - x) ** 2 + (y_circle - y) ** 2 <= radius ** 2

        # Adjust the pixel values within the circular mask based on left or right click
        if event.button == 1:  # Left click
            simulated_predictions[mask] = np.minimum(1.0, simulated_predictions[mask] + 1.)
        elif event.button == 3:  # Right click
            simulated_predictions[mask] = np.maximum(0.0, simulated_predictions[mask] - 1.)

        # Clip values to ensure they don't exceed [0, 1]
        simulated_predictions = np.clip(simulated_predictions, 0, 1)

        # Update the displayed images and AUC-ROC curve
        calculate_roc_auc(ground_truth, simulated_predictions, axes)


# Set the size of the maps
map_size = (100, 100)

# Generate initial simulated predictions starting at 0
simulated_predictions = np.zeros(map_size)

# Create a fixed circular ground truth
initial_radius = 35
center = (map_size[1] // 2, map_size[0] // 2)
ground_truth = generate_circular_ground_truth(map_size, center, initial_radius)

# Plot the initial maps with a rainbow colormap and scaled to [0, 1]
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

axes[0].imshow(ground_truth, cmap='rainbow', interpolation='nearest')
axes[0].set_title('Ground Truth')

axes[1].imshow(simulated_predictions, cmap='rainbow', interpolation='nearest')
axes[1].set_title('Simulated Predictions')

axes[2].set_title('Receiver Operating Characteristic (ROC) Curve')

# Connect the mouse click event to the onclick function
fig.canvas.mpl_connect('button_press_event', onclick)

# Display the plot
plt.show()
