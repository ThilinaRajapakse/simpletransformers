rm -r cache_dir
rm -r outputs
python train.py electra-small

rm -r cache_dir
rm -r outputs
python train.py electra-base

rm -r cache_dir
rm -r outputs
python train.py bert

rm -r cache_dir
rm -r outputs
python train.py roberta

rm -r cache_dir
rm -r outputs
python train.py distilbert

rm -r cache_dir
rm -r outputs
python train.py distilroberta

rm -r cache_dir
rm -r outputs
python train.py xlnet
