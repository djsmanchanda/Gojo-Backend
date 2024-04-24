import os
from ultralytics import YOLO

# Set the path to your dataset folder
dataset_path = "C:\\Users\\djsma\\Downloads\\radixDataset"

# Set the paths to the test, train, and validate folders
test_path = os.path.join(dataset_path, 'test')
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'validate')

# Create a new YOLO model
model = YOLO('yolov8n.pt')  # You can choose a different YOLOv8 variant if desired

# Configure the model
model.train(data=dataset_path, epochs=100, imgsz=640, batch=8, name='yolov8n_alzheimers')

# Save the trained model
model_path = 'trained_model.pt'
model.save(model_path)

# Function to make predictions on new images
def predict_alzheimers(image_path):
    # Load the trained model
    loaded_model = YOLO(model_path)
    
    # Perform inference on the image
    results = loaded_model(image_path)
    
    # Extract the predicted bounding boxes and class labels
    boxes = results[0].boxes.xyxy.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()
    
    return boxes, labels

# Example usage of the prediction function
image_path = 'path/to/new/image.jpg'
predicted_boxes, predicted_labels = predict_alzheimers(image_path)

# Print the predicted bounding boxes and labels
print("Predicted Bounding Boxes:")
print(predicted_boxes)
print("Predicted Labels:")
print(predicted_labels)