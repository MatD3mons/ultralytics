from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data='RPC/data.yaml',#data='COCO1/data.yaml
   imgsz=640,
   epochs=10,
   batch=6,
   task='oneshot',
   name='yolov8n_RPC',
   val=True,
   device=[0]
   )
