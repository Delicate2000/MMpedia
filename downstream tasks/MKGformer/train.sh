device="cuda:2"
batch_size=48
python train.py --epochs=12 --device=$device --batch_size=$batch_size
python train.py --noise=True --epochs=12 --device=$device --batch_size=$batch_size
