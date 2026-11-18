from ultralytics import YOLO

# Load your trained model
model = YOLO('path/to/runs/detect/train/weights/best.pt')

# Use the model to predict on an image or video
results = model.predict('path/to/your/image.jpg')
results.show()  # Show detected results
