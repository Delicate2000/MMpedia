# Entity information collection

To facilitate running downstream tasks, we package the intersection of MMpedia and DB15K.

For BERT-based you can run the model by following script
```
bash train_text.sh # BERT
bash train_our.sh # BERT+ResNet50+Our
bash train_noise.sh # BERT+ResNet50+Noise
bash train_vilt_our.sh # ViLT+Our
bash train_vilt_noise.sh # ViLT+Noise
```

For other methods you can run the model by following script
```
bash train.sh # Entity prediction
bash train_rel.sh # Relation prediction
```

For KG-BERT, first download the source code are from (https://github.com/yao8839836/kg-bert/) then replace the file "run_bert_link_prediction.py" and "run_bert_relation_prediction.py"

For RotatE and ComplEX, first download the source code from (https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/) then replace the folder "code"

For LineaRE, first download the source code from (https://github.com/pengyanhui/LineaRE/) then replace the folder "config" and the file "run.py"

For MKGformer, we refer to (https://github.com/zjunlp/MKGformer/)

For MoSE+RSME, download the source code from (https://github.com/OreOZhao/MoSE4MKGC/) and put or replace the code in folder "src". Note that this project first need to employ BERT and VIT to extract features.