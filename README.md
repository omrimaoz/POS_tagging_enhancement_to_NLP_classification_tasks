# NLP_Project

POS tagging enhancement to NLP classification tasks
Omri Maoz, Tal Levi
Tel-Aviv University

## Instruction

1. Install python version 3.7 or later.
2. clone the entire code and datasets from main branch.
3. Install python packages from requierments.txt (On a virtual environment would be preferable).
4. run "python nltk_download.py" in terminal and download 'Collection->popular', 'Corpora->wordnet' and 'Models->punkt'in the opened window.
5. unzip IMDB_Dataset_v1_csv.zip and News_Dataset_v1_json.zip 
6. run "python prepare_datasets_IMDB.py" and "python prepare_datasets_NEWS.py" in the terminal from the 'Datasets' folder to produce compatible datasets  (this step might take time). Or first, unzip "IMDB_Dataset_15000_json.zip" and "News_Dataset_15000_json.zip" and then run both terminal lines to make the process shorter.
7. run "python main --dataset_name {dataset_name} --tag_feature {tag_feature} --dataset_size {dataset_size} --model {model}" in terminal from the root folder. replace each input from the bank:
- dataset_name ∈ ['IMDB', 'News']
- tag_feature ∈ ['original', 'upos', 'upos_filter', 'upos_filter_extend', 'bigram', 'bigram', 'bigram_filter', 'bigram_filter_extend']
- dataset_size ∈ [15000, 5000, 1000]
- model ∈ ['BiLSTM', 'DAN', 'Transformer]
