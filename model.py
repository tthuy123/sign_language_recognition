from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn import metrics
import os
import numpy as np
import csv

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('TRAIN_DATASET') 

# Actions that we try to detect
# Đọc danh sách actions từ file gloss.txt
with open('./dataset/easy2.txt', 'r') as f:
    actions = np.array([line.strip() for line in f if line.strip()])

# Videos are going to be 50 frames in length
sequence_length = 50

# Gán nhãn cho từng action
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        continue
    for sequence in os.listdir(action_path):
        try:
            sequence_path = os.path.join(action_path, sequence)
            window = []
            for frame_num in range(sequence_length):
                frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
                if not os.path.exists(frame_path):
                    raise FileNotFoundError(f"Missing frame: {frame_path}")
                res = np.load(frame_path)
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
        except Exception as e:
            print(f"Skipping sequence {sequence} in action '{action}': {e}")

# Convert landmarks and labels to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Build the model
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model
model = Sequential()

# LSTM 1
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(50, 1662)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# LSTM 2
model.add(LSTM(256, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.4))

# LSTM 3
model.add(LSTM(128, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Dense layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

# Output layer
model.add(Dense(10, activation='softmax'))  # 100 classes

model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['categorical_accuracy'])


# Train model
model.fit(X_train, y_train, epochs=200, validation_split=0.1, callbacks=[tb_callback, early_stop])

# Lưu model
model.save('action_kdn2.h5')

# Make predictions on the test set
predictions = np.argmax(model.predict(X_test), axis=1)
test_labels = np.argmax(y_test, axis=1)

accuracy = metrics.accuracy_score(test_labels, predictions)
print("Test Accuracy after loading model:", accuracy)

# with open('accuracy_result.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Test Accuracy'])
#     writer.writerow([accuracy])
