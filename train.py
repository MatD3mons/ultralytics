from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8s.pt')
 
# Training.
results = model.train(
   data='RPC/data.yaml',
   imgsz=640,
   epochs=100,
   batch=3,
   task='oneshot',
   name='yolov8s_RPC',
   val=True,
   )