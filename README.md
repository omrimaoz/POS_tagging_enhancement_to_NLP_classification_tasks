# NLP_Project

POS tagging enhancement to NLP classification tasks
Omri Maoz, Tal Levi
Tel-Aviv University

## Instruction

1. Install python version 3.7 or later.
2. clone the entire code and datsets from main branch.
3. Install python packages from requierments.txt (On a virtual enviorment would be preferrable).
4. run "python nltk_download.py" in terminal and download 'Collection->popular' and 'Corpora->wordnet' in the opened window.
5. run "python prepare_datasets_IMDB.py" and "python prepare_datasets_NEWS.py" in the terminal from the 'Datasets' folder to produce compatiable datasets  (this step might takes time).
6. run "python main --dataset_name {dataset_name} --tag_feature {tag_feature} --dataset_size {dataset_size} --model {model}" in terminal from the root folder. replace each input from the bank:
- dataset_name ∈ ['IMDB', 'News']
- tag_feature ∈ ['original', 'upos', 'upos_filter', 'upos_filter_extend', 'bigram', 'bigram', 'bigram_filter', 'bigram_filter_extend']
- dataset_size ∈ [15000, 5000, 1000]
- model ∈ ['BiLSTM', 'DAN', 'Transformer]
