import transformers

MAX_LEN=512
EPOCHS=2
TRAIN_BATCH_SIZE=128
VALID_BATCH_SIZE=64
MODEL_PATH='../input/bert_base_uncased/'
TRAIN_PATH_1='../input/jigsaw-toxic-comment-train.csv'
TRAIN_PATH_2='../input/jigsaw-unintended-bias-train.csv'
VALID_PATH='../input/validation.csv'
SUBMISSION_PATH='../input/submission.csv'
transformer=transformers.BertTokenizer.from_pretrained(
    '../input/bert_base_uncased', 
    do_lower_case=True
)