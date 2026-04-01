import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ---------- PATH SETUP ----------
PROJECT_ROOT = r"D:\FYP\PlantGrowth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- LOAD MODELS ----------
# Stage classifier
stage_model = torch.load(
    "models/stage_classifier.pth",
    map_location=DEVICE
)
stage_model.eval()

# Fruit detector
detector = YOLO("runs/detect/laboro_fast_high_map50/weights/best.pt")

# Growth regressor
class GrowthRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)

growth_model = GrowthRegressor().to(DEVICE)
growth_model.load_state_dict(
    torch.load("models/growth_regressor.pth", map_location=DEVICE)
)
growth_model.eval()

# ---------- RIPENESS (HSV) ----------
def hsv_ripeness(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    h, s, _ = cv2.split(hsv)

    red = ((h < 10) | (h > 160)) & (s > 80)
    green = (h > 35) & (h < 85) & (s > 60)

    if np.sum(red) / h.size > 0.25:
        return 1.0
    elif np.sum(green) / h.size > 0.25:
        return 0.0
    else:
        return 0.5

# ---------- MAIN ----------
img_path = "test.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---- Stage prediction ----
stage_tensor = torch.tensor(
    img_rgb / 255.0, dtype=torch.float32
).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    stage_score = torch.softmax(stage_model(stage_tensor), dim=1)[0, 1].item()

# ---- Fruit detection ----
results = detector.predict(img_rgb, conf=0.4, device=0)[0]

ripeness_scores = []

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img_rgb[y1:y2, x1:x2]

    if crop.size == 0:
        continue

    r = hsv_ripeness(crop)
    ripeness_scores.append(r)

    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

# ---- Growth logic ----
if len(ripeness_scores) == 0:
    ripeness_score = 0.0
else:
    ripeness_score = float(np.mean(ripeness_scores))

X = torch.tensor(
    [[stage_score, ripeness_score]],
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():
    growth = growth_model(X).item()

# ---- Display ----
if len(ripeness_scores) == 0:
    print(f"Growth Percentage (no fruit): {growth:.2f}%")
else:
    print(f"Growth Percentage (with fruit): {growth:.2f}%")
    cv2.putText(
        img,
        f"Growth: {growth:.1f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 0, 0),
        3
    )
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
