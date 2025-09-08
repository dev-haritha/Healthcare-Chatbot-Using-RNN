import random
import json
import torch
import math
import geocoder
import nltk
from model_chat import RNNModel  # Import the RNN model
from nltk_utils import bag_of_words, tokenize
nltk.download('punkt_tab')
# Ensure NLTK 'punkt' resource is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data_rnn.pth"  # Update this file name
data = torch.load(FILE,weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
num_layers = data["num_layers"]  # Add the number of layers used in your RNN model
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = RNNModel(input_size, hidden_size, output_size, num_layers).to(device)
model.load_state_dict(model_state)
model.eval()
fallback_responses = [
    "I'm not sure about that. Could you provide more details?",
    "I couldn't confidently determine your condition. Could you rephrase?",
    "That's unclear. Can you describe your symptoms in a different way?",
    "I don't have enough information. Please try again with more details."
]

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    if ("name" in sentence) or ("this is" in msg.lower()):
        for wor in sentence:    
            if (wor.lower() != "my") and (wor.lower() != "is") and (wor.lower() != "name") and (wor.lower() != "i") and (wor.lower() != "am") and (wor.lower() != "this"):
                user_name = wor.capitalize()
                res = "Hi " + user_name + " please say your age."
                return ["name", res]
    if ("age" in sentence) or ("i" in sentence and "am" in sentence) or ("i'm" in sentence):
        for wor in sentence:
            if wor.isnumeric():
                user_age = wor
                res = "What is your gender?"
                return ["age", res]
    if ("male" in sentence) or ("female" in sentence) or ("Male" in sentence) or ("Female" in sentence):
        for wor in sentence:
            if (wor.lower() == "male") or (wor.lower() == "female"):
                user_gender = wor.lower()
                res = "Tell the symptoms you have to know about potential conditions."
                return ["gender", res]
    if ("yes" in sentence) or (("medical" in sentence) and "center" in sentence) or ("hospital" in sentence) or ("hospitals" in sentence):
        li = centres()
        return li

    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X).unsqueeze(0).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() < 0.95:  # Increased threshold
        return ["not_understand",  random.choice(fallback_responses)]

    for intent in intents['intents']:
        if intent["tag"] == tag:
            if tag == "Pneumothorax" and prob.item() < 0.95:
                return ["not_understand", "I couldn't confidently determine your condition."]
            
            if tag in ["greeting", "goodbye", "work", "who", "Thanks", "joke", "name", "age", "gender"]:
                return [intent['tag'], intent['responses']]
            
            return [intent['tag'], intent['responses'], intent['Precaution']]

    # ✅ Default fallback for unknown inputs
    return ["not_understand", "I'm sorry, I couldn't confidently determine your condition. "]

    
def centres():
    # Function to calculate the Haversine distance between two points
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0  # Radius of the Earth in kilometers

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance

    given_location = [17.7754, 83.3363]  # Fixed location for Kommadi

    with open("medical_centers.json", "r") as json_file:
        medical_centers = json.load(json_file)

    distances_to_centers = []

    for center in medical_centers["intents"]:
        center_location = center["location"]
        distance = haversine(given_location[0], given_location[1], center_location[0], center_location[1])
        distances_to_centers.append((center["tag"], distance))

    distances_to_centers.sort(key=lambda x: x[1])

    l = ["center"]

    for i, (center_name, distance) in enumerate(distances_to_centers[:5], start=1):
        for center in medical_centers["intents"]:
            if center["tag"] == center_name:
                l.append([center_name, (str(round(distance, 2))+'km'), center["Address"]])
    return l

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print("Bot:", resp)
