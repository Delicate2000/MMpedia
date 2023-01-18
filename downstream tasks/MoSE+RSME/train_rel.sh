note="rel"
note2="rel_noise"

device='cuda:3'
one_model='ComplExMDR'
max_epochs=100
rsme_alpha=0.3

model_path1="./ckpt/${one_model}/${note}_img_.pth"
python learn_rel.py --model=$one_model --fusion_img=True --note=$note --img_info=../data/DB15K/img_vec.pickle --early_stopping=10 --max_epochs=$max_epochs --device=$device
python meta_learner_rel.py --model_path=$model_path1 --device=$device


model_path2="./ckpt/${one_model}/${note2}_img_.pth"
python learn_rel.py --model=$one_model --fusion_img=True --note=$note2 --img_info=../data/DB15K/img_vec_noise.pickle --early_stopping=10 --max_epochs=$max_epochs --device=$device
python meta_learner_rel.py --model_path=$model_path2 --device=$device


one_model="RSME"
python learn_rel.py --model=$one_model --early_stopping=10 --fusion_dscp=False --max_epoch=100  --note=$note --device=$device --rsme_alpha=$rsme_alpha --rank=1000
python learn_rel.py --model=$one_model --fusion_dscp=False --max_epoch=100  --img_info=../data/DB15K/img_vec_noise.pickle --note=$note2 --early_stopping=10 --device=$device --rsme_alpha=$rsme_alpha --rank=1000