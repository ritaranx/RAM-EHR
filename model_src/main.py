
import argparse
import os
from utils import load_and_cache_examples, init_logger, load_tokenizer
from trainer import Trainer
import torch 
import numpy as np 
import random 


def set_seed(args):
    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if args.n_gpu > 0  and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)
        torch.cuda.manual_seed(args.train_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset, num_labels, train_size  = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset, num_labels,  dev_size = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset, num_labels, test_size = load_and_cache_examples(args, tokenizer, mode="test")
    # unlabeled_dataset, unlabeled_size = load_and_cache_unlabeled_examples(args, tokenizer, mode = 'unlabeled', train_size = train_size)
   
    print('number of labels:', num_labels)
    print('train_size:', train_size)
    print('dev_size:', dev_size)
    print('test_size:', test_size)
    trainer = Trainer(args, train_dataset=train_dataset, dev_dataset=dev_dataset,test_dataset=test_dataset, \
            unlabeled = None, \
            num_labels = num_labels, data_size = train_size
            )
   

    trainer.init_model()
    trainer.train(n_sample = len(train_dataset))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0,1,2,3', type=str, help="which gpu to use")
    parser.add_argument("--n_gpu", default=1, type=int, help="which gpu to use")

    parser.add_argument("--seed", default=0, type=int, help="which seed to use")
    parser.add_argument("--train_seed", default=0, type=int, help="which seed to use")
    parser.add_argument("--task", default="agnews", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="../datasets", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation script, result directory")
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--dev_file", default="dev.tsv", type=str, help="dev file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--unlabel_file", default="unlabeled.tsv", type=str, help="Test file")

    parser.add_argument("--dev_labels", default=100, type=int, help="number of labels for dev set")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.",)

    parser.add_argument('--logging_steps', type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--model_type", default="bert-base-uncased", type=str)
    parser.add_argument("--tokenizer", default="bert-base-uncased", type=str)

    parser.add_argument("--auto_load", default=1, type=int, help="Auto loading the model or not")
    parser.add_argument("--add_sep_token", action="store_true", help="Add [SEP] token at the end of the sentence")

    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=100, type=int, help="Training steps for initialization.")
    parser.add_argument("--weight_decay", default=1e-3, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--self_training_batch_size", default=32, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--eval_batch_size", default=256, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--max_seq_len_test", default=128, type=int, help="The maximum total input sequence length after tokenization.")

    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument('--gen_model', type = str, default = 'gen_model', help = 'the model used for generating training data.')

    parser.add_argument('--multi_label', type = int, default = 0, help = 'whether train with multi label or not')
    parser.add_argument('--use_def', type = int, default = 1, help = 'whether use definition')
    parser.add_argument('--method_name', type = str, default = 'name', help = 'whether train with multi label or not')


    # semi_method
    args = parser.parse_args()
    if args.model_type == 'pubmedbert':
        args.model_name_or_path = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
        args.tokenizer = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
    elif 'pubmedbert-large' in args.model_type:
        args.model_name_or_path = 'microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract'
        args.tokenizer = 'microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract'
    elif 'distill-clinicalbert' in args.model_type:
        args.model_name_or_path = 'nlpie/clinical-distilbert-i2b2-2010'
        args.tokenizer = 'nlpie/clinical-distilbert-i2b2-2010'
    elif 'mobile-clinicalbert' in args.model_type:
        args.model_name_or_path = 'nlpie/clinical-mobilebert-i2b2-2010'
        args.tokenizer = 'nlpie/clinical-mobilebert-i2b2-2010'
    else:
        args.model_name_or_path = args.model_type
        args.tokenizer = args.model_type
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    if args.task in ['mimic']:
        args.num_labels = 25
    elif args.task in ['cradle']:
        args.num_labels = 1
    main(args)