import transformers

# paths
DATA_PATH = "../data/"
MODEL_DIR = "../data/bert_base_uncased/"
MODEL_PATH = "../data/bert_base_uncased/pytorch_model.bin"

# hyper-params
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 4
EPOCHS = 10
ACCUMULATION = 2
TOKENIZER =  transformers.BertTokenizer.from_pretrained(MODEL_PATH, do_lower_case=True)



