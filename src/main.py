##################################################
#         python script for PinYin input         #
#              written by NLDXTD                 #
##################################################

import re
import sys
import json
import pickle
import argparse
import itertools
from os import listdir
from os.path import isfile, join
from functools import partial
from collections import defaultdict

def parseArg():
    parser = argparse.ArgumentParser(description='Input Args')

    # IOs
    parser.add_argument('--verbose', action='store_true', help="Whether to print more information")
    parser.add_argument('--input-file', type=str, default='../data/input.txt', help="Path of file that will be translated")
    parser.add_argument('--output-file', type=str, default='../data/output.txt', help="Path of file that will be written into")

    # Paths
    parser.add_argument('--load-model', type=str, default='../model/3model.pkl', help="Path of model that will be loaded")
    parser.add_argument('--save-model', type=str, default='../model/model.pkl', help="Path of model that will be saved")
    parser.add_argument('--words', type=str, default='../model/words.pkl', help="Path of word-table that will be saved")
    parser.add_argument('--pinyin-table', type=str, default='../model/pinyin_table.pkl', help="Path of pinyin-table that will be saved")

    # Tasks
    parser.add_argument('--init-words', action='store_true', help="Task: init legal words table")
    parser.add_argument('--init-pinyin-table', action='store_true', help="Task: init pinyin-table")
    parser.add_argument('--train', action='store_true', help="Task: train model")
    parser.add_argument('--translate', action='store_true', help="Task: translate sentence")

    # Files
    parser.add_argument('--encoding', type=str, default='utf8', help="Coding method of input files")
    parser.add_argument('--file', type=str, default=None, help="Path of file to train from")
    parser.add_argument('--dir', type=str, default=None, help="Path of directory to train from")

    # Models Param
    parser.add_argument('--n-gram', type=int, default=3, help="using n-gram model")

    args = parser.parse_args()
    return args

def load(path, encoding, verbose):

    '''Load file from {path} and verbose if needed'''
    if verbose:
        print(f'Loading {path}')
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    if verbose:
        print(f'\rLoaded {path} ')
    return lines

def save_pickle(data, path, verbose=True):

    '''Save computed data into pickle'''
    if verbose:
        print(f'Saving {path}')
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    if verbose:
        print(f'\rSaved {path} ')

def return_param(param):

    '''Return anything'''
    return param

def create_ndic(param, n=1):

    '''Create n times dic with param as default'''
    dic = partial(return_param, param)
    '''Dic recieve a var and return the param'''
    for _ in range(n):
        dic = partial(defaultdict, dic)
    return dic()

def words_dic(args):
    '''create a dic to test whether a word fo exist'''
    word_exist = create_ndic(False)
    words = load(args.file, args.encoding, args.verbose)[0]
    for word in words:
        word_exist[word] = True
    save_pickle(word_exist, args.words)

def init_pinyin_table(args):
    '''Create the mapping from pinyin to words'''
    '''If some kinds of pinyin does not appear in the pinyin-table, it will return a list of '@' '''
    pinyin2words = create_ndic(['@'])  # For @-clipping magic

    lines = load(args.file, args.encoding, args.verbose)
    for line in lines:
        pinyin, *words = line.strip().split(' ')  # e.g. line = "a 啊 嗄 腌 吖 阿 锕"
        pinyin2words[pinyin] = words # pinyin = 'a', words = ['啊', '嗄', '腌', '吖', '阿', '锕']

    save_pickle(pinyin2words, args.pinyin_table)

def load_pickle(path, verbose):

    '''Load pickle from path'''
    if verbose:
        print(f'Loading {path}')
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if verbose:
        print(f'\rLoaded {path} ')
    return data

def get_text(line, word_exist):

    '''Get clean text filled with @ from line'''
    data = json.loads(line)
    words = '@'.join((data['html'], data['title']))
    # Replace all unknown words with '@'
    words = ''.join(
        word if word_exist[word] else '@'
        for word in words
    )
    words = f'@{words}@'
    # Remove duplicate '@'
    words = re.sub(r'(@)\1+', r'\1', words)
    return words

def get_useable_key(prefix):
    '''
    e.g. prefix = "@习访美后@特将出席"
    return "特将出席"
    '''
    if prefix[-1] == '@':
        return prefix[-1]
    return prefix.split('@')[-1]

def train_model(args, model, word_exist):

    '''Training model'''
    def train(path):
        
        def record(prefix, word):
            model[prefix][word] += 1

            if len(prefix) > 1:
                model[prefix]['total'] += 1  # Already recorded for len = 1
                record(prefix[1:], word)

        '''Training data from path'''
        print(f'training from {path}')
        with open(path, 'r', encoding=args.encoding) as f:
            for index, line in enumerate(f):
                # First get text from the line of f
                words = get_text(line, word_exist)
                words_length = len(words)
                for word in words:
                    model[word]['total'] += 1
                model['all']['total'] += words_length

                n_pairs = (
                    words[s:s+args.n_gram]
                    for s in range(words_length - args.n_gram + 1)
                )
                for n_pair in n_pairs:
                    prefix, word = n_pair[:-1], n_pair[-1]
                    prefix = get_useable_key(prefix)
                    record(prefix, word)
            
            return

    '''Training models from single file'''
    if args.file is not None:
        if args.verbose:
            print(f"Training files is:")
            print(f'\t{args.file}')
        train(args.file)
        return

    '''Train models from directory'''
    files = [
        join(args.dir, f)
        for f in listdir(args.dir)
        if f != '../train/.DS_Store'
    ]

    if args.verbose:
        print(f"Training files are:")
        for f in files:
            print(f'\t{f}')

    for f in files:
        train(f)

def translate(args, model, py2word, instream=sys.stdin, outstream=sys.stdout):

    def calc_prob(prefix, word):
        
        def single_calc(prefix, word):
            '''calc prob of word after prefix'''
            eps = 1e-8 
            prob = 0
            if prefix in model:
                prob += 0.95 * model[prefix].get(word, 0) / (model[prefix]['total'] + eps)
            return prob + 0.05 * model[word]['total'] / (model['all']['total'] + eps)

        '''Calculate \cigma P#(word | prefix)'''
        pre_length = args.n_gram - 1
        prefix = get_useable_key(prefix[-pre_length:])
        prob = (
            single_calc(prefix[-i:], word) 
            for i in range(1, len(prefix)+1)
        )
        return sum(prob)

    def find_max(pairs):

        '''Find pair with the maximum probability'''
        def prob_of_pair(pair):
            (text, prob) = pair
            return prob

        return max(pairs, key=prob_of_pair)

    def dp_search(words_list):

        '''Dp search the right answer for pinyins'''

        def combine_of(end_list):
            *pre_list, last_words = end_list
            return (
                (prefix + (last_word, ) for prefix in itertools.product(*pre_list))
                for last_word in last_words
            )

        def calcprob(text, words, prob):
            '''calculate the probability of words accure after text'''

            if text == "":
                text, words = words[0], words[1:]
                prob = model[text]['total']
            
            for word in words:
                prob *= calc_prob(text, word)
                text += word

            return prob

        def prob_of_pair(pair):
            (text, prob) = pair
            return prob

        prefix_length = args.n_gram - 1
        pre_list, end_list = words_list[:-prefix_length], words_list[-prefix_length:]

        if len(words_list) <= prefix_length:
            pre_pairs = [("", None)]
        else:
            pre_pairs = dp_search(pre_list)

        prefix = []
        for post_str in combine_of(end_list):
            tp_pairs = (
                (text+"".join(words), calcprob(text, words, prob))
                for (text, prob), words in itertools.product(pre_pairs, post_str)
            )
            prefix.append(find_max(tp_pairs))
        return prefix

    for pinyinline in instream:

        pinyinline = pinyinline.strip().lower()
        pinyinline = f'@ {pinyinline} @'
        pinyins = pinyinline.split(' ')
        words_list = [py2word[pinyin] for pinyin in pinyins]

        '''using dp to translate'''
        pairs = dp_search(words_list)
        text, prob = find_max(pairs)
        text = text[1:-1]

        print(f'{text}', file=outstream)

def main():
    args = parseArg()

    if args.init_words:

        '''Create a table to store the legal words'''
        words_dic(args)
        return

    if args.init_pinyin_table:

        '''Create a table to store the pinyin-character list'''
        init_pinyin_table(args)
        return

    if args.train:

        '''Training n_grams model base on the training data'''
        word_exist = load_pickle(args.words, args.verbose)
        train_model(args, model, word_exist)
        save_pickle(model, args.save_model)
        return

    model = load_pickle(args.load_model, args.verbose)
    py2word = load_pickle(args.pinyin_table, args.verbose)
    io = {}
    if args.input_file is not None:
        io['instream'] = open(args.input_file, 'r', encoding=args.encoding)
    if args.output_file is not None:
        io['outstream'] = open(args.output_file, 'w') 
    translate(args, model, py2word, **io)

    for _, f in io.items():

        '''Close the files'''
        f.close()
    
    return

if __name__ == '__main__':
    main()