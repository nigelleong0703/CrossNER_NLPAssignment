@echo off
python main.py --exp_name science_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm science
python main.py --exp_name ai_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm ai
python main.py --exp_name literature_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm literature
python main.py --exp_name music_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm music
python main.py --exp_name politics_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm politics