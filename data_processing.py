import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt  # Import for visualization
from imblearn.over_sampling import RandomOverSampler  # For data augmentation


# Load the dataset
data = pd.read_csv('fer2013.csv')

# Display some info
print("Original data shape:", data.shape)
print(data.head())

# Check class distribution
emotion_counts = data['emotion'].value_counts()
print("\nEmotion Class Distribution:\n", emotion_counts)

# Visualize emotion distribution
plt.figure(figsize=(10, 6))
emotion_counts.sort_index().plot(kind='bar')
plt.title('Distribution of Emotions')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('emotion_distribution.png')  # Save the plot
plt.show()


# Handle missing or corrupted data
print("\nChecking for missing data...")
print(data.isnull().sum())  # Check for NaN values
data = data.dropna() # Remove rows with NaN values (if any)

# Prepare data
X = []
y = []
for index, row in data.iterrows():
    try:
        pixels = np.array(row['pixels'].split(), dtype='float32')
        image = pixels.reshape(48, 48)
        X.append(image)
        y.append(row['emotion'])
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        continue  # Skip to the next row

X = np.array(X)
y = np.array(y)

# Normalize data
X = X / 255.0

# Balance the dataset using oversampling
print("\nBalancing the dataset using RandomOverSampler...")
oversampler = RandomOverSampler(random_state=42)
X_reshaped = X.reshape(X.shape[0], -1)  # Reshape for oversampling
X_resampled, y_resampled = oversampler.fit_resample(X_reshaped, y)
X_resampled = X_resampled.reshape(-1, 48, 48)  # Reshape back

print("Shape of X before split:", X_resampled.shape)
print("Shape of y before split:", y_resampled.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# Save processed data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("\nData preprocessing and balancing complete!")
