import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

from models import EfficientNetB0, SimpleEfficientNetB0
from tools import calc_data_mean_std, evaluate_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


try:
    with open("data_mean_std.json", "r", -1, "utf-8") as f:
        data_desc = json.loads("".join(f.readlines()))
except FileNotFoundError:
    data_desc = calc_data_mean_std("data")

data_mean = data_desc["mean"]
data_std = data_desc["std"]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std),
])

test_dataset = datasets.ImageFolder("data/test", transform=transform)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

classes = len(test_dataset.classes)  # 10

if __name__ == "__main__":
    teacher_model = EfficientNetB0(classes).to(device)
    teacher_model.load_state_dict(
        torch.load(
            "models_status/10c_model_epoch_7.pth",
            map_location=device
            )
        )

    student_model = SimpleEfficientNetB0(classes).to(device)
    student_model.load_state_dict(
        torch.load(
            "models_status/10c_student_model_epoch_7.pth",
            map_location=device
            )
        )

    t_accuracy = evaluate_model(test_loader, teacher_model, device)
    print(f"Teacher Accuracy: {t_accuracy:.2f}%")

    s_accuracy = evaluate_model(test_loader, student_model, device)
    print(f"Student Accuracy: {s_accuracy:.2f}%")