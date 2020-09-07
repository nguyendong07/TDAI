from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="C:/Users/ABC/Desktop/New folder/TDAI/Data/hololens")
trainer.evaluateModel(model_path="C:/Users/ABC/Desktop/New folder/TDAI/Data/hololens/models", json_path="C:/Users/ABC/Desktop/New folder/TDAI/Data/hololens/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
