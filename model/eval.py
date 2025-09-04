import torch

from model.model import AcademicStressDetector
import const.parameters as parameters
from utils.dataset import input_size
from utils.dataset import scaler


model = AcademicStressDetector(
    input_size=input_size, 
    hidden_size=parameters.hidden_size,
    num_classes=parameters.num_classes,
    dropout=parameters.dropout,
    gamma=parameters.gamma,
    lr=parameters.learning_rate
)

model.load_state_dict(torch.load("./models/model.pth"))
model.eval()

inputDict = {
    "Have you been getting headaches more often than usual?" : 0,                 
    "Do you face any sleep problems or difficulties falling asleep?" : 0,               
    "Have you noticed a rapid heartbeat or palpitations?" : 0,    
    "Is your working environment unpleasant or stressful?" : 0,   
    "Is your hostel or home environment causing you difficulties?" : 0,
    "Do you feel safe in your environment?" : 0,
    "Do you have access to basic needs (food, shelter, etc.)?" : 0,
    "Do you lack confidence in your academic performance?" : 0,
    "Do you feel overwhelmed with your academic workload?" : 0,
    "Are you facing any difficulties with your professors or instructors?" : 0,
    "Do you lack confidence in your choice of academic subjects?" : 0,
    "Do you get support from friends, family, or peers?" : 0,
    "Are you in competition with your peers, and does it affect you?" : 0,
    "Are academic and extracurricular activities conflicting for you?" : 0,
    "Have you experienced bullying?" : 0
}

categories = ["Eustress", "Distress", "No Stress"]

for key in inputDict.keys():
    property = int(input(f"{key} [0-5]: "))
    inputDict[key] = property

x_manual = torch.tensor([list(inputDict.values())], dtype=torch.float32)

x_manual_scaled = scaler.transform(x_manual)
x_manual_scaled = torch.tensor(x_manual_scaled, dtype=torch.float32)

with torch.no_grad():
    logits = model(x_manual_scaled)
    prediction = torch.argmax(logits, dim=1).item()
    print(f"Prediction: {categories[prediction]}")