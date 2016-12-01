# coding: utf-8
import pandas as pd
import numpy as np
import math
import logging
import sys
import re
import time
import json
import string
import jieba
import requests
import time
#import clustering_functions
from . import clustering_functions
import subprocess
import multiprocessing
from collections import Counter
from itertools import count

logpath = '/home/lingtelli/jlsche/croton/api/logs'

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s', filename=logpath)
logger = logging.getLogger('logtest')

def startClustering(comments, origin_comments, tfidf_df, roles):
    comments_df = comments.to_frame('comment')
    comments_df['origin_comment'] = origin_comments
    comments_df['seg_comment'] = comments_df['comment'].apply(clustering_functions.generateSegmentList)

    # 列出所有的斷詞，以及其出現次數
    word_counted = Counter([word for comment in comments_df['seg_comment'] for word in comment])

    # 刪除seg_word內不必要的東西
    seg_words = list(word_counted.keys())
    useless_words = ['[','!','吧','还','那么','也','什么','好','真的','是','你','我','他','的','了','啦','很','太','哦','/','r','n','吗','呢','呀','哎','她','噢','么','耶','还是','最','啊',']', ' ']

    clean_seg_words = [w for w in seg_words if w not in useless_words]

    # 找出所有clean_seg_words的tf-idf value，以dataframe的方式去取得tf-idf value因為比較快
    seg_word_df = pd.DataFrame()
    seg_word_df['word'] = clean_seg_words
    seg_word_df['tf_idf'] = seg_word_df['word'].apply(clustering_functions.getTfIdfValue, args=(tfidf_df,))

    # 把所有斷詞的 tf-idf dataframe 變成 dict, 方便未來查詢用
    tfidf_dict = dict(zip(seg_word_df['word'], seg_word_df['tf_idf']))

    # 紀錄相同評論的次數，之後相同的評論只要處理一次
    comments_count = comments_df.groupby('origin_comment').size().reset_index(name='count')
    comments_df.drop_duplicates(subset='origin_comment', inplace=True)
    comments_df = comments_count.merge(comments_df, how='right', on='origin_comment')

    if roles is not None:
        comments_df['center'] = comments_df['seg_comment'].apply(clustering_functions.findRole, args=(roles, ))
    else:
        comments_df['center'] = False

    mask = (comments_df['center'] == False)
    comments_df_valid = comments_df[mask]

    # method1: center is the word with highest tf-idf value
    #comments_df.loc[mask, 'center'] = comments_df_valid['seg_comment'].apply(clustering_functions.getCenterCandidate_TfIdf, args=(tfidf_dict, ))

    # method2: center is the word with most common word in group
    comments_df.loc[mask, 'center'] = comments_df_valid['seg_comment'].apply(clustering_functions.getCenterCandidate_MostCommon, args=(word_counted, useless_words, ))

    # 去除掉評論內的center
    comments_df['clean_seg_comment'] = comments_df.apply(clustering_functions.removeCenter2, axis=1)
    comments_df['clean_seg_comment'] = comments_df['clean_seg_comment'].apply(lambda x: x[0])

    # 根據tf-idf值，找出每一句評論的關鍵詞(critical_word)
    comments_df['critical_word'] = comments_df['clean_seg_comment'].apply(clustering_functions.getCriticalWord, args=(tfidf_dict,))

    # 找出critical word的關聯詞找出來，並轉為簡體 (api需輸入簡體 輸出為繁體)
    words_for_query = list(set(comments_df['critical_word'].tolist()))
    headers = {'Content-type': 'application/json', }

    resp_assoc = None
    with requests.Session() as s:
        s.keep_alive = False
        #resp_assoc = requests.post('http://lingtelli.com:5012/getKeywordAssoc/', data=json.dumps({"keyword": words_for_query}), headers=headers)
        resp_assoc = requests.post('http://localhost:3005/getKeywordAssoc?lang=sim', data={'keyword': comments_df['critical_word'].tolist()})
        #resp_assoc = requests.post('http://192.168.10.108:3005/getKeywordAssoc?lang=sim', data={'keyword': comments_df['critical_word'].tolist()}) 
    '''
    concept_tree = widgets.buildConceptTree(resp_assoc)
    equivalent_words, parent = widgets.findEquivalent(concept_tree)

    critical_words = comments_df['critical_word'].unique()
    print(critical_words)
    root_of = dict()
    for w in critical_words:
        root = widgets.findRoot(w, concept_tree, equivalent_words)
        if w != root:
            root_of[w] = root
    
    #comments_df['root_of_critical_word'] = comments_df['critical_word'].apply(widgets.findRoot, args=(concept_tree, equivalent_words, parent))
    '''
    critical_words_with_assoc = clustering_functions.getWordsWithAssoc(resp_assoc)
    comments_df['words_with_assoc'] = comments_df['critical_word'].apply(clustering_functions.getAssoc, args=(critical_words_with_assoc,))

    # 看assoc_words_set裡的關聯詞，其在評論中出現的次數並排列

    # 取得所有評論的關連詞的聯集
    assoc_words_set = set([item for sublist in comments_df['words_with_assoc'] for item in sublist])
    element_count = {key: 0 for key in assoc_words_set if key not in useless_words}

    # 因為群中心是由LCS產生，故有可能關連詞會有像是群中心的詞，如李大仁
    for words, count in zip(comments_df['words_with_assoc'], comments_df['count']):
        for word in words:
            if word not in useless_words:
                element_count[word] += count
    sorted_element_count = sorted(element_count.items(), key=lambda x: x[1])
    sorted_element_count.reverse()

    # 從每一個評論的關連詞(comments_df['word_with_assoc'])裡，找出哪個關連詞最先在sorted_element_count出現（兩陣列中找到第一個共同元素）
    # http://stackoverflow.com/questions/16118621/first-common-element-from-two-lists
    # 此方法會有評論無法被分群，故需要以choosen來紀錄
    cluster = {}
    sorted_element_count_top = [a for a in sorted_element_count if a[1] > 1]

    comments_df['first_assoc_word'] = comments_df['words_with_assoc'].apply(clustering_functions.findFirstAssocWord, args=(sorted_element_count_top, ))
    #comments_df['first_assoc_word'] = comments_df['words_with_assoc'].apply(clustering_functions.findFirstAssocWord, args=(sorted_element_count_top, root_of))

    result_df_list = []
    unique_center = comments_df['center'].unique()
    unique_assoc_word = comments_df['first_assoc_word'].unique()
    for center in unique_center:
        grouped_center = comments_df[comments_df['center']==center]
        unique_assoc_word = grouped_center['first_assoc_word'].unique()

        for word in unique_assoc_word:
            grouped = grouped_center[grouped_center['first_assoc_word']==word]
            grouped['copied_origin_comment'] = grouped.apply(clustering_functions.copyMore, axis=1)
            total_count = grouped.loc[:, 'count'].sum()

            sum_comments = grouped['copied_origin_comment'].tolist()
            comments_str = ',  '.join(sum_comments)

            group_df = pd.DataFrame(columns=['comment','cluster_center','keyword','member_size'], index=[0])
            group_df['cluster_center'] = center
            group_df['keyword'] = word
            group_df['comment'] = comments_str
            group_df['member_size'] = total_count
            result_df_list.append(group_df)

    try:
        result = pd.concat(result_df_list, ignore_index=True)
        return result
    except:
        print('nothing to concate for this cluster.')
        return None
    return None

def parallelClustering(comments_list, origin_comments_list, tfidf_df, roles):
    results = []
    for comments, origin_comments in zip(comments_list, origin_comments_list):
        results.append(startClustering(pd.Series(comments), pd.Series(origin_comments), tfidf_df, roles))
    return results


def main(dir_id):
    #data_path = '/home/lingtelli/croton/croton/console/data/' + dir_id + '/'
    #current_path = '/home/lingtelli/jlsche/croton/clustering'
    data_path = '/home/ubuntu/web/croton/croton/console/data/' + dir_id + '/'
    current_path = '/home/ubuntu/croton_clustering/'
    #data_path = '/Users/jlin/Lingtelli/croton/working_version/data/' + dir_id + '/'
    #current_path = '/Users/jlin/Lingtelli/croton/working_version/'
    filename = 'cluster_result0.csv'
    input_file = data_path + filename

    logger.info("Receive request %s." % dir_id)

    try:
        subprocess.check_output(['grep', '-nr', "\r", "%s" % input_file])
        backup_filename = 'cluster_result0_noCR.csv'
        subprocess.check_call(['sed', '-e', "s/\r//g", "%s" % (input_file)], stdout=open(data_path + backup_filename, "wb"))
        filename = backup_filename
        print('CR detected, erasing ... done.')
    except:
        logger.exception("Task %s has no pov file." % dir_id)
        print('NO CR IN FILE')

    print('reading role.csv from directory', dir_id, '... ', end='')
    try:
        roles = pd.read_csv(data_path + 'role.csv', squeeze=True)
        for r in roles:
            jieba.add_word(r)
        print('done.')
    except:
        roles = None
        logger.exception("Task %s has no role file." % dir_id)
        print('\nNo role.csv in', dir_id, 'directory.')

    df = pd.read_csv(data_path + filename, names=['count', 'comments'])
    df['count'] = df['count'].apply(lambda x: int(x))
    df = df[df['count'] > 0]
    
    ### 應該先清除一些髒東西, \n \r 目前只會變成 nr
    df['cleaned_comments'] = df['comments'].apply(lambda x: str(x).replace('\n', '').replace('\r', ''))
    df['cleaned_comments'] = df['cleaned_comments'].apply(clustering_functions.removePunctuation)
    tfidf_df = pd.read_csv(current_path + 'data/sim_word_lex_utf8_2.pl')

    comments_series = df['comments'].apply(clustering_functions.strToSeries)
    cleaned_comments_series = df['cleaned_comments'].apply(lambda x: x.split(',  '))
    
    task_size = len(comments_series)
    print('Task Size:', task_size)

    # number of cores is:
    # 12 on machine 10.108
    # 4 on 118.192.8.106 and my MBA
    NUM_OF_CORES = 12

    print('# of cores:', NUM_OF_CORES)
    p = multiprocessing.Pool(NUM_OF_CORES)
    results = []
    job_each_core = []
    percentage_each_core = [0.001, 0.009, 0.01, 0.01, 0.01, 0.01, 0.05, 0.1, 0.1, 0.2, 0.2, 0.3]

    lower = 0
    for i in percentage_each_core:
        upper = lower + math.ceil(task_size * i)
        job_each_core.append((lower, upper))
        lower = upper
       
    origin_clusters = comments_series.tolist()
    cleaned_clusters = cleaned_comments_series.tolist()
    
    start_time = time.time()
    for arg in job_each_core:
        results.append(p.apply_async(parallelClustering, args=(cleaned_clusters[arg[0]: arg[1]], origin_clusters[arg[0]: arg[1]], tfidf_df, roles, )))
    
    results = [r.get() for r in results]
    
    results = [x for r in results for x in r if x is not None]
    output_df = pd.concat(results)
    output_df.to_csv(data_path + 'multiprocessing_temp.csv', index=False, sep=',', encoding='utf8')
    end_time = time.time()
    print("%.2f seconds to finish the task %s" % ((end_time - start_time), dir_id))

    ########################################################################
    ### not sure if lines below could solve the problem
    ########################################################################
    logger.info("Finish clustering task %s. size: %d, time: %.2f" % (dir_id, task_size, end_time-start_time))

    try:
        p.terminate()
    except Exception:
        logger.exception("Finish clustering task %s, but something went wrong when terminating Pool." % dir_id)
        print("Something went wrong when terminating Pool.")
        sys.exit()

    return 'OK'

if __name__ == "__main__":
    dir_id = sys.argv[1]
    main(str(dir_id))
  
