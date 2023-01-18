device="cuda:1"
epochs=5
batch_size=24
python train_rel.py --epochs=$epochs --device=$device --batch_size=$batch_size
python train_rel.py --noise=True --epochs=$epochs --device=$device --batch_size=$batch_size
