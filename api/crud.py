import torch
from .models import StressInput
from model.model import AcademicStressDetector
import const.parameters as parameters
from utils.dataset import input_size, scaler

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

categories = ["Eustress", "Distress", "No Stress"]

def get_stress_level_prediction(data: StressInput) -> str:
    x_manual = torch.tensor([list(data.dict().values())], dtype=torch.float32)
    
    x_manual_scaled = scaler.transform(x_manual)
    x_manual_scaled = torch.tensor(x_manual_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x_manual_scaled)
        prediction_idx = torch.argmax(logits, dim=1).item()
    
    return categories[prediction_idx]