mkdir data
wget https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz -P data
tar -xvf data/paws_wiki_labeled_final.tar.gz -C data
mv data/final/* data
rm -r data/final

wget http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv -P data
