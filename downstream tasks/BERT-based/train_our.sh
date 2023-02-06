device="cuda:2"
epochs=13
lr=0.00002
batch_size=48
python model_word.py --image_type=Our  --with_image --epochs=$epochs --device=$device --do_train --task=ph --lr=$lr --batch_size=$batch_size

python model_word.py --image_type=Our  --with_image --epochs=$epochs --device=$device --do_train --task=pt --lr=$lr --batch_size=$batch_size

