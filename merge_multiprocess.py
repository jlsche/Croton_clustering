import os
import sys
import logging
import pandas as pd

logpath = '/home/ubuntu/croton_clustering/logs' 
#logpath = '/Users/jlin/Lingtelli/croton/logs' 

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s', filename=logpath)
logger = logging.getLogger('logtest')

def grouping(x):
    x['comment'] = x['comment'].apply(lambda x: str(x))
    return pd.Series(dict(member_size = x['member_size'].sum(), comment = "%s" % ',  '.join(x['comment'])))

def splitNegative(comments, *args):
    negative = args[0]
    seg = comments.split(',  ')
    positive_comments = []
    negative_comments = []

    for s in seg:
        if any(x in s for x in negative):
            negative_comments.append(s)
        else:
            positive_comments.append(s)
    
    negative_comments_str = ',  '.join(negative_comments)
    positive_comments_str = ',  '.join(positive_comments)
    
    return positive_comments_str + 'split_me' + negative_comments_str

def getClusterSize(comments):
    splitted = comments.split(',  ')
    return 0 if len(splitted[0]) == 0 else len(splitted)

    

def merge_all_df(dir_id):
    #input_path = '/home/lingtelli/jlsche/croton/temp_cluster/'
    #output_path = '/home/lingtelli/croton/croton/console/data/' + dir_id + '/'
    data_path = '/home/ubuntu/web/croton/croton_v2/console/data/' + dir_id + '/'
    #data_path = '/Users/jlin/Lingtelli/croton/working_version/data/' + dir_id + '/'
    
    df = pd.read_csv(data_path + 'multiprocessing_temp.csv')

    #df = df[(df['keyword']!='NO_ASSOC_WORD') & (df['keyword']!='NO_CRITICAL_WORD') & (df['cluster_center']!='OOPS')]
    grouped = df.groupby(['cluster_center','keyword']).apply(grouping).sort_values(by=['member_size'], ascending=[False])
    #grouped = grouped[grouped['member_size']>=3]

    grouped = grouped[['member_size', 'comment']]           # grouped = grouped.reset_index() if you want to output all field 
    grouped = grouped.reset_index()

    logger.info("Start merging task %s." % dir_id)

    pov_flag = False
    try:
        negative_words = pd.read_csv(data_path + 'pov.csv', header=0, squeeze=True)
        negative_words = negative_words.tolist()
        pov_flag = True
        print('pov file found.')
    except:
        print('\nNo pov.csv in', dir_id, 'directory.')
        grouped = grouped[grouped['member_size'] > 0]
        grouped = grouped.sort_values(by=['member_size'], ascending=False)
        grouped.to_csv(data_path + 'cluster_result.csv', columns=['member_size', 'comment'], header=False, index=False, encoding='utf8')
        logger.exception("Task %s has no pov file." % dir_id)
        print('done.')

    if pov_flag is True:
        grouped['sorted_comment'] = grouped['comment'].apply(splitNegative, args=(negative_words, ))
        s = grouped.sorted_comment.str.split('split_me').apply(pd.Series, 1).stack()
        s.index = s.index.droplevel(-1)
        s.name = 'sorted_comment'
        del grouped['sorted_comment']
        grouped = grouped.join(s)
        grouped['member_size'] = grouped['sorted_comment'].apply(getClusterSize)##lambda x: len(x.split(',  ')))
        grouped = grouped[grouped['member_size'] > 0]
        grouped = grouped.sort_values(by=['member_size'], ascending=False)
        grouped.to_csv(data_path + 'cluster_result.csv', columns=['member_size', 'sorted_comment', 'keyword', 'cluster_center'], header=False, index=False, encoding='utf8')
        print('done.')

    logger.info("Merging task %s, done." % dir_id)
    return 'OK'
    
    # logfile 的路徑
    # datapath的路徑
    # api的網址

    #grouped.to_csv(output_path + 'try2_cluster_result.csv', columns=['member_size', 'comment'], header=False, index=False, encoding='utf8')

if __name__ == '__main__':
    dir_id = sys.argv[1]
    merge_all_df(str(dir_id))
