from ultralytics import YOLO

# Load a model
model = YOLO('runs/oneshot/yolov8n_RPC/weights/best.pt')

print(model.info())
# Validate the model
metrics = model.val( data='RPC/data.yaml',
   imgsz=640,
   batch=8,
   task='oneshot',
   name='yolov8n_RPC_valid',)  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category