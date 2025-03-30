import os
import matplotlib.pyplot as plt

# Define the paths to the subdirectories
data_dir = "data"
subdirs = {
    "train_eyes/closed": os.path.join(data_dir, "train_eyes", "closed"),
    "train_eyes/open": os.path.join(data_dir, "train_eyes", "open"),
    "valid_eyes/closed": os.path.join(data_dir, "valid_eyes", "closed"),
    "valid_eyes/open": os.path.join(data_dir, "valid_eyes", "open"),
}

# Function to count images in a directory
def count_images(directory):
    return len([file for file in os.listdir(directory) if file.endswith(('.png', '.jpg', '.jpeg'))])

# Count images before augmentation
before_augmentation_counts = {}
for section, path in subdirs.items():
    if os.path.exists(path):
        before_augmentation_counts[section] = count_images(path)
    else:
        before_augmentation_counts[section] = 0

# Simulate augmentation (replace this with actual augmentation logic if needed)
# For now, assume augmentation has already been performed.

# Count images after augmentation
after_augmentation_counts = {}
for section, path in subdirs.items():
    if os.path.exists(path):
        after_augmentation_counts[section] = count_images(path)
    else:
        after_augmentation_counts[section] = 0

# Print the results
print("Number of images before augmentation:")
for section, count in before_augmentation_counts.items():
    print(f"{section}: {count} images")

print("\nNumber of images after augmentation:")
for section, count in after_augmentation_counts.items():
    print(f"{section}: {count} images")

# Plot the results as a bar chart
sections = list(before_augmentation_counts.keys())
before_counts = list(before_augmentation_counts.values())
after_counts = list(after_augmentation_counts.values())

x = range(len(sections))  # X-axis positions

plt.figure(figsize=(12, 6))
plt.bar(x, before_counts, width=0.4, label='Before Augmentation', color='blue', align='center')
plt.bar([i + 0.4 for i in x], after_counts, width=0.4, label='After Augmentation', color='green', align='center')
plt.xlabel('Sections')
plt.ylabel('Number of Images')
plt.title('Comparison of Images Before and After Augmentation')
plt.xticks([i + 0.2 for i in x], sections, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()