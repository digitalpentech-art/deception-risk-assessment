import csv
import os
import random

def generate_csv_data():
    # Faces: dummy csv representing pixels
    os.makedirs('data/raw/faces', exist_ok=True)
    for i in range(5):
        with open(f'data/raw/faces/face_{i}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # Create a 224x224 grid of random pixel values (0-255)
            for _ in range(224):
                writer.writerow([random.randint(0, 255) for _ in range(224)])

    # Speech/Voice/Micro: similarly simple CSVs
    os.makedirs('data/raw/speech', exist_ok=True)
    for i in range(5):
        with open(f'data/raw/speech/speech_{i}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([random.random() for _ in range(100)]) # 100 features

    os.makedirs('data/raw/voice', exist_ok=True)
    for i in range(5):
        with open(f'data/raw/voice/voice_{i}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([random.random() for _ in range(15)]) # 15 features

    os.makedirs('data/raw/micro', exist_ok=True)
    for i in range(5):
        with open(f'data/raw/micro/micro_{i}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            # 10 frames of flat data
            for _ in range(10):
                writer.writerow([random.random() for _ in range(50)])

    print("Dummy CSV data generated successfully.")

if __name__ == '__main__':
    generate_csv_data()
