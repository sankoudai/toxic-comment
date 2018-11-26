#coding=utf8
# 重要特征词抽取: Gini系数
from sklearn.feature_extraction.text import CountVectorizer

# label
df = pd.read_csv('data/train.csv')
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df['label'] = df[label_cols].max(axis=1)

# P
P = {}
P[0] = len(df[df['label']==0]) / len(df)
P[1] = len(df[df['label']==1]) / len(df)

# p
p = {}
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
word_cnt = count_vect.fit_transform(df['comment_text'])
words = count_vect.get_feature_names()
for i in range(len(df)):
    row = word_cnt.getrow(i)
    label = df['label'][i]
    for word_ind in row.indices:
        if word_ind >= len(words):continue
        if word_ind not in p: 
            p[word_ind] = [0,0,0]
        p[word_ind][label] += 1
        p[word_ind][2] += 1

# gini
ginis = []
for word_ind in p:
    norm_p0 = (p[0]/p[2]) / P[0]
    norm_p1 = (p[1]/p[2]) / P[1]
    gini = (norm_p0 / (norm_p0 + norm_p1))**2 + (norm_p1 / (norm_p0 + norm_p1))**2
    
    word = words[word_ind]
    if pw[0] > 10:
        continue
    if (norm_p1< 0.50) or len(word)<=2:
        continue
    if word.isdigit():
        continue
    ginis.append(word, gini)

ginis = sorted(ginis, key=lambda x:-x[1])

with open('data/badwords.txt', 'w') as g:
    line = 'word\tgini\n'
    g.write(line)
    for word, gini in ginis:
        line = '{}\t{}\n'.format(word, gini)
        g.write(line)
