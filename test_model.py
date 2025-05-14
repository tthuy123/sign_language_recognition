import os
import numpy as np
import csv
from sklearn import metrics
from tensorflow.keras.models import load_model

SEQUENCE_LENGTH = 50
GLOSS_PATH = './dataset/easy2.txt'
MODEL_PATH = 'action_kdn2.h5'
TEST_DATA_PATH = 'TEST_DATASET'
OUTPUT_CSV = 'test_performance.csv'

with open(GLOSS_PATH, 'r') as f:
    actions = np.array([line.strip() for line in f if line.strip()])

label_map = {label: num for num, label in enumerate(actions)}

# ===== 3. Load dữ liệu từ TEST_DATASET =====
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(TEST_DATA_PATH, action)
    if not os.path.exists(action_path):
        continue

    for sequence in os.listdir(action_path):
        try:
            sequence_path = os.path.join(action_path, sequence)
            window = []
            for frame_num in range(SEQUENCE_LENGTH):
                frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
                if not os.path.exists(frame_path):
                    raise FileNotFoundError(f"Missing frame: {frame_path}")
                res = np.load(frame_path)
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
        except Exception as e:
            print(f"Skipping sequence {sequence} in action '{action}': {e}")

X_test = np.array(sequences)
y_test = np.array(labels)

print(f"Tổng số mẫu test: {len(X_test)}")

model = load_model(MODEL_PATH)

y_probs = model.predict(X_test)

# Top-1 Accuracy
y_pred_top1 = np.argmax(y_probs, axis=1)
top1_acc = metrics.accuracy_score(y_test, y_pred_top1)

# Top-5 Accuracy
top5_correct = sum(y_test[i] in np.argsort(y_probs[i])[::-1][:5] for i in range(len(y_test)))
top5_acc = top5_correct / len(y_test)

# Top-10 Accuracy
top10_correct = sum(y_test[i] in np.argsort(y_probs[i])[::-1][:10] for i in range(len(y_test)))
top10_acc = top10_correct / len(y_test)
# ===== 6. Kết quả =====
print(f"Top-1 Accuracy: {top1_acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")
print(f"Top-10 Accuracy: {top10_acc:.4f}")
# ===== 7. Ghi ra file CSV =====
# with open(OUTPUT_CSV, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Top-1 Accuracy', 'Top-5 Accuracy'])
#     writer.writerow([top1_acc, top5_acc])

# print(f"Kết quả đã được lưu vào: {OUTPUT_CSV}")
