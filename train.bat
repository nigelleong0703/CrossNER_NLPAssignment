@echo off
@REM python main.py --exp_name science_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm science
@REM python main.py --exp_name ai_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm ai
@REM python main.py --exp_name literature_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm literature
@REM python main.py --exp_name music_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm music
@REM python main.py --exp_name politics_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm politics


python main.py --exp_name ai_split_pretrain_then_finetunee --exp_id 1 --num_tag 35 --batch_size 16 --newbert --conll --tgt_dm ai