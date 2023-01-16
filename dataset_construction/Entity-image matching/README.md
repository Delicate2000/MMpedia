# Entity-image matching
We need multi-process acceleration to perform calculations for many entities. CLIP cannot directly apply package multiprocess (https://github.com/openai/CLIP/issues/130), we deploy it on the framework of gunicorn+flask.

First, run following command to build server
```
gunicorn --workers=7 --timeout 120 --threads 5 clip_flask:app -b 0.0.0.0:5001 -k gevent
```
Then run the code block in jupyter multiprocess clip
