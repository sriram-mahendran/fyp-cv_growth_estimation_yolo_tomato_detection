import os
import cv2
from torch.utils.data import Dataset

class StageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        # 🔒 ABSOLUTE PATH SAFETY
        root_dir = os.path.abspath(root_dir)

        print("[StageDataset] Using root_dir:", root_dir)

        classes = {
            "vegetative": 0,
            "flowering": 1
        }

        for cls_name, label in classes.items():
            cls_dir = os.path.abspath(os.path.join(root_dir, cls_name))

            print(f"[StageDataset] Checking folder: {cls_dir}")

            if not os.path.exists(cls_dir):
                raise FileNotFoundError(
                    f"StageDataset ERROR: folder not found → {cls_dir}"
                )

            for file in os.listdir(cls_dir):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(cls_dir, file), label)
                    )

        if len(self.samples) == 0:
            raise RuntimeError("StageDataset ERROR: No images found!")

        print(f"[StageDataset] Total images loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path)
        if image is None:
            # skip bad image
            return self.__getitem__((idx + 1) % len(self.samples))


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
