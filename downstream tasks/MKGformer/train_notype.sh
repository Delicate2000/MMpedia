device="cuda:0"
batch_size=48
python train.py --no_type=True --epochs=12 --device=$device --batch_size=$batch_size
