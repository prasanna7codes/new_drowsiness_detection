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

# Count and display the number of images in each subdirectory
image_counts = {}
for section, path in subdirs.items():
    if os.path.exists(path):
        image_counts[section] = count_images(path)
    else:
        image_counts[section] = 0

# Print the results
for section, count in image_counts.items():
    print(f"{section}: {count} images")

# Plot the results as a bar chart
sections = list(image_counts.keys())
counts = list(image_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(sections, counts, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Sections')
plt.ylabel('Number of Images')
plt.title('Number of Images in Each Section')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()