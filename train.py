from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='COCO1/data.yaml',#data='COCO1/data.yaml
   imgsz=640,
   epochs=200,
   batch=120,
   task='oneshot',
   name='yolov8n_COCO1_support2',
   val=True,
   device=[5,6,7]
   )