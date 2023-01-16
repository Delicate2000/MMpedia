# Entity-image matching

First, put vgg_flask.py and yolo_flask.py into under the root directory of their source project
YOLOv5:(https://github.com/ultralytics/yolov5)
VGG19:(https://github.com/hjptriplebee/VGG19_with_tensorflow)

Then run following command to build server
```
gunicorn --workers=7 --timeout 120 --threads 5 yolo_flask:app -b 0.0.0.0:5000 -k gevent
gunicorn --workers=7 --timeout 120 --threads 5 vgg_flask:app -b 0.0.0.0:5001 -k gevent
```

Finally run the code block in jupyter yolo+vgg
