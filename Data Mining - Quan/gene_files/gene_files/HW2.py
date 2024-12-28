# Step 1: Load Data Without Using Pandas or Other Libraries
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split by comma (or adjust if delimiter is different)
            row = line.strip().split(',')
            # Replace missing values with a placeholder
            row = [value if value != '' else 'Unknown' for value in row]
            data.append(row)
    return data

# Load train and test data
train_path = 'C:\Users\nguye\Desktop\IS6482- Data Mining\Data Mining - Quan\gene_files\gene_files\Genes_relation.data'
test_path = 'Genes_relation.test'
train_data = load_data(train_path)
test_data = load_data(test_path)

# Define column index for target variable (assume "Localization" is the last column)
target_index = len(train_data[0]) - 1

# Drop "Function" column (assuming it's the second last column)
for row in train_data:
    row.pop(-2)
for row in test_data:
    row.pop(-2)

# Step 2: Convert Categorical Variables to Numeric Codes
def encode_data(data):
    # Create a dictionary to store mappings
    mappings = [{} for _ in range(len(data[0]))]
    encoded_data = []
    
    for row in data:
        encoded_row = []
        for i, value in enumerate(row):
            if value not in mappings[i]:
                mappings[i][value] = len(mappings[i])
            encoded_row.append(mappings[i][value])
        encoded_data.append(encoded_row)
    
    return encoded_data, mappings

encoded_train_data, mappings = encode_data(train_data)
encoded_test_data, _ = encode_data(test_data)

# Step 3: Implement Naive Bayes from Scratch
def calculate_priors(data, target_index):
    priors = {}
    total_count = len(data)
    for row in data:
        target_value = row[target_index]
        if target_value not in priors:
            priors[target_value] = 0
        priors[target_value] += 1
    for key in priors:
        priors[key] /= total_count
    return priors

def calculate_likelihoods(data, target_index):
    likelihoods = {}
    target_levels = set(row[target_index] for row in data)
    feature_count = len(data[0]) - 1
    
    for target in target_levels:
        target_data = [row for row in data if row[target_index] == target]
        feature_probs = [{} for _ in range(feature_count)]
        
        for row in target_data:
            for i in range(feature_count):
                feature_value = row[i]
                if feature_value not in feature_probs[i]:
                    feature_probs[i][feature_value] = 0
                feature_probs[i][feature_value] += 1
        # Normalize probabilities
        for i in range(feature_count):
            total = len(target_data)
            for key in feature_probs[i]:
                feature_probs[i][key] /= total
        likelihoods[target] = feature_probs
    
    return likelihoods

def predict_naive_bayes(data, priors, likelihoods, target_index):
    predictions = []
    for row in data:
        class_probs = priors.copy()
        for target in class_probs.keys():
            for i, feature_value in enumerate(row):
                if i != target_index:
                    # Apply Laplace smoothing with a small probability
                    class_probs[target] *= likelihoods[target][i].get(feature_value, 1e-6)
        predictions.append(max(class_probs, key=class_probs.get))
    return predictions

# Train Naive Bayes
priors = calculate_priors(encoded_train_data, target_index)
likelihoods = calculate_likelihoods(encoded_train_data, target_index)
predictions = predict_naive_bayes(encoded_test_data, priors, likelihoods, target_index)

# Step 4: Calculate Accuracy
actuals = [row[target_index] for row in encoded_test_data]
correct = sum(1 for pred, actual in zip(predictions, actuals) if pred == actual)
accuracy = correct / len(actuals)
print(f"Accuracy: {accuracy:.4f}")
