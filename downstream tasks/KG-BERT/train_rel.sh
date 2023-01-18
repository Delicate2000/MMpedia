output_dir="./output_DB15K_rel/"
python run_bert_relation_prediction.py --task_name kg  --do_train  --do_eval --do_predict  --max_seq_length 150 --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 20.0 --output_dir $output_dir  --gradient_accumulation_steps 1 --eval_batch_size 32
