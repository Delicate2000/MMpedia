import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import flask
from flask import request, jsonify
import clip
import torch
from PIL import Image
import json

app = flask.Flask(__name__)
#manager=Manager(app)

clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

def concat_pictures(entity_name, path):
    imgs = []
    files = []
    num = 0

    for base_url,_,file_names in os.walk(path + entity_name):
        for file_name in sorted(file_names):
            files.append(path + entity_name + '/' + file_name)
            image1 = preprocess(Image.open(os.path.join(base_url,file_name))).unsqueeze(0)
            imgs.append(image1)
            del(image1)
            num += 1
    if imgs == []:
        image_list = []
        image_list = torch.tensor(image_list)
    else:
        image_list = torch.cat((imgs), 0)
    return image_list, files


# 得到clip 图片与文本匹配分数
def clip_predict(image_list, abstract):
    if len(image_list) == 0:
        return []
    
    with torch.no_grad():
        image_list = image_list.to('cuda')
        text = clip.tokenize([abstract], truncate=True)
        text = text.to('cuda')
        
        logits_per_image, logits_per_text = clip_model(image_list, text)
        probs = logits_per_text.softmax(dim=-1)
        best_score = max(logits_per_text[0])
        best_index = torch.argmax(probs).item()
        best_prob = max(probs[0])
    return logits_per_text,best_index




@app.route('/test',methods=['GET'])
def predict():
    try:
        entityname = request.args.get('entityname')
        path = request.args.get('path')  # receive parameters

        img_list, files = concat_pictures(entityname, path)
        text = request.args.get('text')  # data process

        results, best_index = clip_predict(img_list, text)
        results = results.to('cpu')
        results = results.detach().numpy().tolist()
        results = list(results)

        ans = {}
        ans['results'] = results
        ans['files'] = files

        del(results)
        del(files)

        return jsonify(ans)
    except Exception as e:
        print(e)
        return 'wrong'



if __name__ == "__main__":      
    app.run(host='127.0.0.1',port=5001)