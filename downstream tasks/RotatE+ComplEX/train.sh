python -u codes/run.py --cuda --do_train  --do_valid  --do_test  --data_path a  --model RotatE  -n 256 -b 1024 -d 1000  -g 24.0 -a 1.0 -adv  -lr 0.0001 --max_steps 40000  -save models/RotatE --test_batch_size 16 -de
python -u codes/run.py --cuda --do_train  --do_valid  --do_test  --data_path a  --model ComplEx  -n 256 -b 1024 -d 1000  -g 24.0 -a 1.0 -adv  -lr 0.0001 --max_steps 40000  -save models/ComplEx --test_batch_size 16 -de --double_entity_embedding --double_relation_embedding

