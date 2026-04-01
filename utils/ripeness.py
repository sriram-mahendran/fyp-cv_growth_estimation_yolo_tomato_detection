import cv2
import numpy as np

def hsv_ripeness(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    red_mask = ((h < 10) | (h > 160)) & (s > 80)
    green_mask = (h > 35) & (h < 85) & (s > 60)

    red_ratio = np.sum(red_mask) / h.size
    green_ratio = np.sum(green_mask) / h.size

    if red_ratio > 0.25:
        return "full"
    elif green_ratio > 0.25:
        return "green"
    else:
        return "half"

def get_ripeness_score(results, image, conf_thresh=0.6):
    votes = {"green": 0.0, "half": 0.0, "full": 0.0}

    for box in results.boxes:
        conf = float(box.conf.item())
        if conf < conf_thresh:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        hsv_vote = hsv_ripeness(crop)
        votes[hsv_vote] += conf

    if sum(votes.values()) == 0:
        return "unknown", votes

    return max(votes, key=votes.get), votes
