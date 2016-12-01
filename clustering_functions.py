import re
import copy
import json
import time
import string
import operator
import jieba
import jieba.analyse
import requests
import multiprocessing

jieba.del_word('...')

def generateSegmentList(comment):
    words = jieba.cut(comment, cut_all=False)
    return list(words)

def strToSeries(string):
    return string.replace(' ', '').split(',')

def getWordsWithAssoc(associations):
    words_with_assoc = {}
    for word in associations.json():
        words_sibling = []
        for word_sibling in word['sibling']:
            words_sibling.append(word_sibling['name'])
        words_with_assoc[word['name']] = word['parents'] + words_sibling + word['children']
    return words_with_assoc

def wordsWithAssocToSim(words_with_assoc):
    sim_words_with_assoc = {}
    with requests.Session() as s:
        s.keep_alive = False
        for word, assoc in words_with_assoc.items():
            sim_words_with_assoc[word] = {}
            sim_words_assoc = []
            for tra_word in words_with_assoc[word]:
                resp = s.get('http://localhost:3001/tra2sim', params={'tra': tra_word})
                sim_words_assoc.append(resp.json()['sim'][0])
            sim_words_with_assoc[word] = sim_words_assoc
        return sim_words_with_assoc

def getAssoc(word, *args):
    if type(word) == list:
        word = word[0]
    words_sim_with_assoc = args[0]
    assoc = []
    try:
        assoc.extend(words_sim_with_assoc[word])
    except:
        print(word, 'not found')
    assoc.append(word)
    return assoc

def getCriticalWord(words, *args):
    tfidf_dict = args[0]
    critical_word_tf_idf_value = -1000.0
    critical_word = None

    for word in words:
        try:
            word_tf_idf_value = tfidf_dict[word]
            if word_tf_idf_value > critical_word_tf_idf_value:
                critical_word_tf_idf_value = word_tf_idf_value
                critical_word = word
        except:
            # word can't find is in remove_list (clean_seg_words)
            pass
    if critical_word == None:
        try:
            critical_word = words[0]
        except:
            critical_word = 'NO_CRITICAL_WORD' + str(time.time())
    return critical_word

def getCenterCandidate_TfIdf(words, *args):
    tf_idf_dict = args[0]
    word_tfidf_dict = {}
    for word in words:
        try:
            word_tfidf_dict[word] = tf_idf_dict[word]
        except:
            word_tfidf_dict[word] = -10.0
    return sorted(word_tfidf_dict.items(), key=operator.itemgetter(1))[-1:][0][0]

def getCenterCandidate_MostCommon(words, *args):
    word_counted = args[0]
    useless_words = args[1]

    for word in word_counted.most_common(30):
        if (word[0] not in useless_words) and (word[0] in words):
            return word[0]
    return 'OOPS' + str(time.time())

def getTfIdfValue(word, *args):
    tf_idf_df = args[0]
    word_tf_idf_value = -1000.0
    try:
        word_index = tf_idf_df[tf_idf_df['sim_word']==word].index[0]
        word_tf_idf_value = tf_idf_df['tf_idf'][word_index]

    except:
        # word not in tf-idf data.
        pass
    return word_tf_idf_value

def removePunctuation(words):
    #'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    #regex1 = re.compile('[%s]' % re.escape(string.punctuation))
    regex = re.compile('[，。”~〈《》〉（）「」『』\n\t；！？～⋯〃《》〃、●■：◎★☆＊※()><=▼【】*．／﹙﹚〔〕＝▓˙@\[\]]')#!\"#$%&\'()*+-./:;<=>?@[\\]^_`{|}~]')
    #return regex2.sub('', regex1.sub('', words))
    return regex.sub('', words)

def findRole(seg_comment, *args):
    roles = args[0]
    first_role = next((word for word in seg_comment if word in roles.tolist()), None)
    return first_role if first_role is not None else False

def removeCenter2(df):
    #if df['center'] not in df['seg_comment']:
    #print(df['center'], ':', type(df['center']), '\t', df['seg_comment'], ':', type(df['seg_comment']), '\tis in?', df['center'] in df['seg_comment'])
    resp = [x for x in df['seg_comment'] if x != df['center']]
    #print(type(df['seg_comment']), type(df['center']), type(resp))
    return [resp]
    resp = [x for x in df if x != center]
    print(center, resp)
    return resp

def findFirstAssocWord(words, *args):
    sorted_word_count_top = args[0]
    set_y = set(words)
    first_assoc_word = next((element for element in sorted_word_count_top if element[0] in set_y), None)
    '''
    root_of = args[1]
    if first_assoc_word is not None:
        print('the first association word is:', first_assoc_word[0])
    if first_assoc_word is not None and root_of.get(first_assoc_word[0]) is not None:
        print(first_assoc_word[0], '->', root_of[first_assoc_word[0]])
        return root_of[first_assoc_word[0]]
    '''
    return first_assoc_word[0] if first_assoc_word is not None else 'NO_ASSOC_WORD_' + str(time.time())

def copyMore(df):
    count = df['count']
    comment = df['origin_comment']
    return ',  '.join([comment[:]] * count)

