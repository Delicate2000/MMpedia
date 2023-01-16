import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # 防止内核挂掉
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import flask
from flask import request, jsonify

import re
import urllib.request
import argparse
import sys
import vgg19
import cv2
import tensorflow as tf
import numpy as np
import caffe_classes
import heapq

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

app = flask.Flask(__name__)

sess = tf.Session()

dropoutPro = 1
classNum = 1000
skip = []

imgMean = np.array([104, 117, 124], np.float)
x = tf.placeholder("float", [1, 224, 224, 3])

model = vgg19.VGG19(x, dropoutPro, classNum, skip)
score = model.fc8
softmax = tf.nn.softmax(score)
# as_default维持一个session
with sess.as_default():
    assert tf.get_default_session() is sess
    sess.run(tf.global_variables_initializer()) # 这句话初始化参数 要放在模型载入前面
    model.loadModel(sess)


def gen_synonyms(path):
    global sess
    withPath = lambda f: '{}/{}'.format(path,f)
    testImg = dict((f,cv2.imread(withPath(f))) for f in sorted(os.listdir(path)) if os.path.isfile(withPath(f)))
    reses = []
    if testImg.values():

        with sess.as_default():
            assert tf.get_default_session() is sess  
            reses = []

            for key,img in testImg.items():
                #img preprocess
                resized = cv2.resize(img.astype(np.float), (224, 224)) - imgMean

                # maxx = np.argmax(sess.run(softmax, feed_dict = {x: resized.reshape((1, 224, 224, 3))}))

                results = sess.run(softmax, feed_dict = {x: resized.reshape((1, 224, 224, 3))}) #[1,1000]向量
                #print("results:", results)

                results = np.array(results)

                maxx = heapq.nlargest(3, range(len(results[0])), results[0].take)

                one_res = []
                res = caffe_classes.class_names[maxx[0]]
                res1 = caffe_classes.class_names[maxx[1]]
                res2 = caffe_classes.class_names[maxx[2]]
                one_res.append(maxx[0])
                one_res.append(maxx[1])
                one_res.append(maxx[2])
                reses.append(one_res)

                font = cv2.FONT_HERSHEY_SIMPLEX

        return reses


@app.route('/test',methods=['GET'])
def predict():
    path = request.args.get('path')
    result = gen_synonyms(path)
    return jsonify(result)

if __name__ == "__main__":      
    app.run(host='127.0.0.1',port=4396)