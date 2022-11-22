import json
import pandas as pd
with open('wtwt_with_text.json', 'r', encoding='utf8')as fp:
    json_data = json.load(fp)
data = pd.DataFrame(json_data)
data = data[~(data.merger == 'FOXA_DIS')]
print(len(data))
label = ['support',  'comment', 'refute','unrelated']
label2id = { polarity:idx for idx, polarity in enumerate(label)}
topics = ['AET_HUM', 'ANTM_CI','CI_ESRX','CVS_AET']
for topic in topics:
    topic_data = data[data.merger == topic]
    # topic_data = topic_data.reset_index()
    topic_data.reset_index()
    print(topic)
    print(len(topic_data))
    o_topic_data = data[~(data.merger == topic)]
    o_topic_data.reset_index()
    # o_topic_data = o_topic_data.reset_index()
    print(len(o_topic_data))
    with open(topic, 'w+',encoding='utf-8') as f:

        for i in range(len(topic_data)):
            item = topic_data.iloc[i]
            f.write(item.text.replace('\n','')+'\n')
            f.write(item.merger+'\n')
            f.write(str(label2id[item.stance])+'\n')
    with open('o_'+topic, 'w+',encoding='utf-8') as f:

        for i in range(len(o_topic_data)):
            item = o_topic_data.iloc[i]
            f.write(item.text.replace('\n','')+'\n')
            f.write(item.merger+'\n')
            f.write(str(label2id[item.stance])+'\n')

