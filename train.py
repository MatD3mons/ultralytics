from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='RPC/data.yaml',
   imgsz=640,
   epochs=100,
   batch=96,
   task='oneshot',
   name='yolov8n_RPC',
   val=True,
   device=[3,4,5]
   )