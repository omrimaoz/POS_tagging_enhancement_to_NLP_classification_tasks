import pandas
import pandas as pd
from datasets import Dataset, DatasetDict


def get_df_from_json(json_path, limit):
    df = pd.read_json(json_path)
    d = df.T
    d = d[:limit]
    return d, d['class'].nunique()



def split_dataset_to_datasetDict(df, train_percent, validate_percent):
    '''
    Parameters
    ----------
    ds - the dataset
    train_percent - will split the dataset into datasets of size train_percent and (1-train_percent)
    validate_percent - will split the training dataset into datasets of size validate_percent and (1-validate_percent)

    Returns
    -------
    datasetDict - datasetDict of the train, val and test datasets
    '''

    n = len(df)
    train_val_size = int(n * train_percent)
    val_size = int(train_val_size * validate_percent)
    train_size = train_val_size - val_size
    test_size = n - train_val_size

    df_train = df[:train_size]
    df_val = df[train_size:train_size + val_size]
    df_test = df[-test_size:]

    ds_train = Dataset.from_pandas(df_train)
    ds_val = Dataset.from_pandas(df_val)
    ds_test = Dataset.from_pandas(df_test)
    return DatasetDict({'train': ds_train,
                        'validate': ds_val,
                        'test': ds_test})


def get_DatasetDict_from_json(json_path):
    df = get_df_from_json(json_path)
    return split_dataset_to_datasetDict(df, 0.8, 0.2)


def get_mode_df(df, mode):
    df_ret = pandas.DataFrame()
    df_ret['text'] = df[mode]
    df_ret['label'] = df['class']

    return df_ret


def get_mode_dataset(df, mode):
    return Dataset.from_pandas(get_mode_df(df, mode))


def load_json(json_path, mode, limit, train_percent=0.8, validate_percent=0.2):
    df, nclass = get_df_from_json(json_path, limit)
    df_m = get_mode_df(df, mode)
    ds_dict = split_dataset_to_datasetDict(df_m, train_percent, validate_percent)

    return ds_dict, nclass
