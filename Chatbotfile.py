import random
import json
import torch
from Chatbotmodel import NeuralNet
from nltkMethods import bag_of_words_data, tokenize_user_message

def getresponse(msg):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('intents_data.json', 'r') as data_intents:
        intents_data = json.load(data_intents)
    
    FILE = "data.pth"
    data_of_file = torch.load(FILE)
    
    input_size = data_of_file["input_size"]
    hidden_size = data_of_file["hidden_size"]
    output_size = data_of_file["output_size"]
    all_words = data_of_file['collect_all_words']
    tags = data_of_file['tags_of_intents']
    used_model_state = data_of_file["used_model_state"]
    
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(used_model_state)
    model.eval()
    
    result=[]
    Emobot_name = "Emobot"
    
    while True:
        message =msg
        if message == "quit":
            break
    
        message = tokenize_user_message(message)
        X = bag_of_words_data(message, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
    
        output = model(X)
        _, predicted = torch.max(output, dim=1)
    
        tag_intent = tags[predicted.item()]
        print(tag_intent)
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for response_intent in intents_data['intents']:
                if tag_intent == response_intent["tag"]:
                    result.append(f"{Emobot_name}: {random.choice(response_intent['responses'])}") 
                    result.append(tag_intent)
                    print(result)
                   
        else:
            result.append(f"{Emobot_name}: {random.choice(response_intent['responses'])}") 
            result.append(tag_intent)
            print(result)
        break
      
               
    return result



