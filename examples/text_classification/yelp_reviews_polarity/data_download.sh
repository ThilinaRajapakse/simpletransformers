mkdir data
wget https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz -O data/data.tgz
tar -xvzf data/data.tgz -C data/
mv data/yelp_review_polarity_csv/* data/
rm -r data/yelp_review_polarity_csv/
rm data/data.tgz