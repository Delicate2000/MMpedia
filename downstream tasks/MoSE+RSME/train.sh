note="11_13"
note2="11_13noise"

device='cuda:0'

one_model='ComplExMDR'
fuse_img=("True" "False")
max_epochs=200
rsme_alpha=0.6

model_path1="./ckpt/${one_model}/${note}_img_.pth"
python learn.py --model=$one_model --fusion_img=True --note=$note --img_info=../data/DB15K/img_vec.pickle --early_stopping=20 --max_epochs=$max_epochs --device=$device
python meta_learner.py --model_path=$model_path1 --device=$device


model_path2="./ckpt/${one_model}/${note2}_img_.pth"
python learn.py --model=$one_model --fusion_img=True --note=$note2 --img_info=../data/DB15K/img_vec_noise.pickle --early_stopping=20 --max_epochs=$max_epochs --device=$device
python meta_learner.py --model_path=$model_path2 --device=$device


one_model="RSME"
python learn.py --model=$one_model --early_stopping=100 --fusion_dscp=False --max_epoch=1000  --note=$note --device=$device --rsme_alpha=$rsme_alpha --rank=1000
python learn.py --model=$one_model --fusion_dscp=False --max_epoch=1000  --img_info=../data/DB15K/img_vec_noise.pickle --note=$note2 --early_stopping=100 --device=$device --rsme_alpha=$rsme_alpha --rank=1000