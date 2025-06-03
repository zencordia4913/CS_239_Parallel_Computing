import numpy as np # Used for matrix operations
import csv # Used for importing csv files


def import_data_set(File_dir):
    # Initialize labels and features to store data
    labels = []
    features = []

    # Read the CSV file
    with open(File_dir, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels.append(float(row[0])) 
            features.append([float(x) for x in row[1:]]) 

    # Convert lists to NumPy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Splitting the dataset indices into training and test indices with equal amount (0.5)
    num_total_samples = len(labels)
    split_index = int(num_total_samples * 0.5)  
    idx_class_minus1 = np.where(labels == -1)[0]
    idx_class_1 = np.where(labels == 1)[0]

    # Calculate number of samples for each class in the split
    num_samples_class_minus1 = int(split_index / 2)
    num_samples_class_1 = split_index - num_samples_class_minus1

    # Shuffle and select indices for each class for training and test sets
    np.random.shuffle(idx_class_minus1)
    np.random.shuffle(idx_class_1)

    train_idx = np.concatenate((idx_class_minus1[:num_samples_class_minus1], idx_class_1[:num_samples_class_1]))
    test_idx = np.concatenate((idx_class_minus1[num_samples_class_minus1:], idx_class_1[num_samples_class_1:]))

    # Split features and labels based on selected indices
    train_data = features[train_idx]
    train_labels = labels[train_idx]

    test_data = features[test_idx]
    test_labels = labels[test_idx]

    return train_data, train_labels, test_data, test_labels




if __name__ == "__main__":
    b_train_data, b_train_labels, b_test_data, b_test_labels = import_data_set(r'C:\Users\Jeryl Salas\OneDrive\Documents\AI 201 Introduction to Ai\PA4\banana_data.csv')
    print(b_train_data)
    print(b_train_labels)
    print(b_test_data)
    print(b_test_labels)




