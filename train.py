from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8s.pt')
 
# Training.
results = model.train(
   data='COCO1/data.yaml',#data='COCO1/data.yaml
   imgsz=640,
   epochs=200,
   batch=90,
   task='oneshot',
   name='yolov8n_COCO1',
   val=True,
   device=[2,3,4]
   )