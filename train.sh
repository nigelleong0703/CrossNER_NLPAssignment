# python main.py --exp_name science_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm science
# python main.py --exp_name ai_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm ai
# python main.py --exp_name literature_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm literature
# python main.py --exp_name music_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm music
# python main.py --exp_name politics_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --conll --tgt_dm politics

# python run_language_modeling.py --output_dir=politics_spanlevel_integrated --model_type=bert --model_name_or_path=bert-base-cased --do_train --train_data_file=corpus/politics_integrated.txt --mlm

python main.py --exp_name ai_split_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --newbert --conll --tgt_dm ai
python main.py --exp_name science_split_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --newbert --conll --tgt_dm science
python main.py --exp_name literature_split_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --newbert --conll --tgt_dm literature
python main.py --exp_name music_split_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --newbert --conll --tgt_dm music
python main.py --exp_name politics__split_pretrain_then_finetune --exp_id 1 --num_tag 35 --batch_size 16 --newbert --conll --tgt_dm politics