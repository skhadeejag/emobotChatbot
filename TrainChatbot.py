import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltkMethods import bag_of_words_data, tokenize_user_message, stem_words
from Chatbotmodel import NeuralNet

'''read json intent file and store'''
with open('Intents_data.json', 'r') as file:
    all_intents = json.load(file)

collect_all_words = []
tags_of_intents = []

'''x input text and y probability of detection '''
xy = []
# loop through each sentence in our intents patterns
for single_intent in all_intents['intents']:
    tag_of_each_intent = single_intent['tag']
    # add to tag list
    tags_of_intents.append(tag_of_each_intent)
    for data_pattern in single_intent['patterns']:
        # tokenize each word in the sentence
        tokenize_msg = tokenize_user_message(data_pattern)
        # add to our words list
        collect_all_words.extend(tokenize_msg)
        # add to xy pair
        xy.append((tokenize_msg, tag_of_each_intent))

# stem and lower each word
ignore_markes = ['?', '.', '!']
collect_all_words = [stem_words(word_) for word_ in collect_all_words if word_ not in ignore_markes]
# remove duplicates and sort
collect_all_words = sorted(set(collect_all_words))
tags_of_intents = sorted(set(tags_of_intents))

print(len(xy), "patterns")
print(len(tags_of_intents), "tags:", tags_of_intents)
print(len(collect_all_words), "unique stemmed words:", collect_all_words)

# create training data
X_train_data = []
y_train_data = []
for (pattern_sentence, tag_intent) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words_data(pattern_sentence, collect_all_words)
    X_train_data.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags_of_intents.index(tag_intent)
    y_train_data.append(label)

X_train_data = np.array(X_train_data)
y_train_data = np.array(y_train_data)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train_data[0])
hidden_size = 8
output_size = len(tags_of_intents)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        
        self.n_samples = len(X_train_data)
        self.x_data = X_train_data
        self.y_data = y_train_data

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader_model = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch_data in range(num_epochs):
    for (extracted_words, labels) in train_loader_model:
        extracted_words = extracted_words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(extracted_words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch_data+1) % 100 == 0:
        print (f'Epoch out of [{epoch_data+1}/{num_epochs}], Loss calculated: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"used_model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"collect_all_words": collect_all_words,
"tags_of_intents": tags_of_intents
}
FILE = "data.pth"
torch.save(data, FILE)

print(f'data training is complete here. file is saved now {FILE}')
