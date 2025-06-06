import cv2
from ultralytics import YOLO
import numpy as np
import math

# Charger YOLO
model = YOLO("yolov8n.pt")

# Ouvrir vidéo
cap = cv2.VideoCapture(r"C:\\Users\\hotsa\\Downloads\\demo1.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # FPS de la vidéo (30 par défaut)

# Simple dictionnaire pour stocker les positions des objets détectés par ID
# Format: {id: {'positions': [(x, y), ...], 'label': 'person' ou 'sports ball'}}
tracked_objects = {}
next_id = 0
max_dist = 50  # distance max pour associer un nouvel objet au même ID (en pixels)

def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    detections = []

    # Récupérer les centres des boîtes et labels pertinents
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label not in ["person", "sports ball"]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        conf = float(box.conf[0])
        detections.append({'pos': (cx, cy), 'box': (x1, y1, x2, y2), 'label': label, 'conf': conf})

    # Association simple : chercher pour chaque détection un ID existant proche
    assigned_ids = []
    for det in detections:
        pos = det['pos']
        label = det['label']

        # Chercher l'ID le plus proche avec même label
        min_id, min_dist = None, max_dist + 1
        for obj_id, data in tracked_objects.items():
            if data['label'] != label:
                continue
            last_pos = data['positions'][-1]
            d = dist(pos, last_pos)
            if d < min_dist:
                min_dist = d
                min_id = obj_id

        if min_dist <= max_dist:
            # On associe cette détection à l'objet existant
            tracked_objects[min_id]['positions'].append(pos)
            assigned_ids.append(min_id)
        else:
            # Nouveau ID
            tracked_objects[next_id] = {'positions': [pos], 'label': label}
            assigned_ids.append(next_id)
            next_id += 1

    # Nettoyer les objets non réassignés (optionnel)
    ids_to_remove = [obj_id for obj_id in tracked_objects if obj_id not in assigned_ids]
    for obj_id in ids_to_remove:
        # Pour garder simple, on supprime les objets non détectés dans cette frame
        del tracked_objects[obj_id]

    # Affichage
    for obj_id, data in tracked_objects.items():
        pos = data['positions'][-1]
        label = data['label']
        x1, y1, x2, y2 = 0, 0, 0, 0
        # On peut retrouver la bbox approximative (pas strict, pour le rectangle)
        # Ici on dessine un cercle sur la position
        color = (0, 255, 0) if label == 'person' else (0, 0, 255)

        # Calcul distance parcourue (en pixels)
        positions = data['positions']
        distance = 0
        for i in range(1, len(positions)):
            distance += dist(positions[i], positions[i-1])

        # Calcul vitesse (pixels/sec)
        vitesse = 0
        if len(positions) > 1:
            vitesse = dist(positions[-1], positions[-2]) * fps

        # Afficher cercle et ID
        cv2.circle(frame, pos, 10, color, 2)
        cv2.putText(frame, f"ID:{obj_id}", (pos[0]+15, pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Afficher distance et vitesse
        cv2.putText(frame, f"D: {distance:.1f}px", (pos[0]+15, pos[1]+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"V: {vitesse:.1f}px/s", (pos[0]+15, pos[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Suivi Joueurs et Ballon", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()