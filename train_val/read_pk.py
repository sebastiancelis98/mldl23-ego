import pickle

# Open the pkl file in read-binary mode
with open('train_val\D1_train.pkl', 'rb') as f:
    # Load the object from the file
    data = pickle.load(f)

print(data)