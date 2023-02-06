device="cuda:1"
epochs=13
lr=0.00002
# lr=0.0001
batch_size=48

python model_word.py --epochs=$epochs --device=$device --do_train --task=ph --lr=$lr --batch_size=$batch_size

python model_word.py --epochs=$epochs --device=$device --do_train --task=pt --lr=$lr --batch_size=$batch_size


