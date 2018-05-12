from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import tensorflow as tf
import re
import numpy as np
import csv
from matplotlib import pyplot
import random
# define training data

WORD_2_VEC_VALUE = 0.97
MIN_COUNT = 3
MAX_COUNT = 1000
GROUP_NUM_MAX = 3

MIN_KEY_TRAIN = 5
MIN_KEY_TEST = 3

NUM_TRAIN = 100
GRADIAN_OPTIMIZER = 0.01

sentences = []
keywords = { }
father = { }

def get_root_father(word) :
    while type(father[word]) is str :
        word = father[word]
    return word

def mix(word_1, word_2):
    word_1 = get_root_father(word_1)
    word_2 = get_root_father(word_2)
    if word_1 == word_2 or father[word_1] + father[word_2] > GROUP_NUM_MAX :
        return
    if father[word_1] < father[word_2] :
        father[word_2] = father[word_1] + father[word_2]
        father[word_1] = word_2
    else :
        father[word_1] = father[word_1] + father[word_2]
        father[word_2] = word_1

def show_father():
    group = { }
    for word in father:
        if type(father[word]) is int and father[word] > 1:
            group[word] = []
    for word in father:
        if type(father[word]) is str :
            tmp = word
            while type(father[tmp]) is str :
                tmp = father[tmp]
            group[tmp].append(word)
    result = []
    for i in group :
        tmp = group[i]
        tmp.append(i)
        result.append(tmp)
    for i in result:
        print(i)


# ---------------------

with open('./data_crawl_article.txt', newline='') as csvfile:
    spamreader = csv.reader(csvfile)

    for row in spamreader:
        # sk = row[0].strip().split(" ")[0]
        sentence = []
        for word in row[0].split(" "):
            w = word.strip().lower()
            if w :
                sentence.append(w)
                if w not in keywords :
                    keywords[w] = 1
                else :
                    keywords[w] += 1
                father[w] = 1
        sentences.append(sentence)

# train model
model = Word2Vec(sentences, min_count=1)

for key in keywords:
    similar = model.wv.similar_by_word(key)
    for s in similar :
        if s[1] > WORD_2_VEC_VALUE :
            mix(key, s[0])

# ----------------------

keynum = {}
skillnum = {}

with open('./data_crawl_article.txt', newline='') as csvfile:
    spamreader = csv.reader(csvfile)

    num = 0
    for row in spamreader:
        for k in row[0].split(" "):
            if k :
                k = get_root_father(k.strip().lower())
                if k not in keynum and keywords[k] > MIN_COUNT and keywords[k] < MAX_COUNT :
                    keynum[k] = num
                    num = num + 1


with open('./data_crawl_skill.txt', newline='') as csvfile:
    spamreader = csv.reader(csvfile)

    num = 0
    for row in spamreader:
        for k in row[0].split(";"):
            k = k.strip()
            if k not in skillnum :
                skillnum[k] = num
                num = num + 1

show_father()
# TRAIN

data = []
x_train = []
y_train = []
x_test = []
y_test = []

with open('./data_crawl_article.txt', newline='') as csvfile:
    spamreader = csv.reader(csvfile)

    for row in spamreader:
        tmp = []
        for i in range(len(keynum)):
            tmp.append(0)
        for i in row[0].split(" "):
            if i :
                i = get_root_father(i.strip().lower())
                if i in keynum :
                    tmp[keynum[i]] = 1
        data.append(tmp)

with open('./data_crawl_skill.txt', newline='') as csvfile:
    spamreader = csv.reader(csvfile)

    for idx,row in enumerate(spamreader):
        tmp = []
        for i in range(len(skillnum)):
            tmp.append(0)
        for k in row[0].split(";"):
            k = k.strip()
            tmp[skillnum[k]] = 1
        data[idx] = { 'x': data[idx], 'y': tmp }

with open('./testing_data.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)

    for idx,row in enumerate(spamreader):
        tmp_x = []
        for i in range(len(keynum)):
            tmp_x.append(0)
        has_key = 0
        for i in row[0].strip().lower().split(" "):
            i = re.sub(r'[\W_]+', '', i)
            if i in father :
                i = get_root_father(i)
                if i in keynum :
                    has_key += 1
                    tmp_x[keynum[i]] = 1

        tmp = []
        for i in range(len(skillnum)):
            tmp.append(0)
        has_skill = 0
        for i in row[1].split(";"):
            i = i.replace('"', '').strip()
            if i in skillnum :
                has_skill += 1
                tmp[skillnum[i]] = 1

        if has_skill > 0 and has_key >= MIN_KEY_TEST:
            x_test.append(tmp_x)
            y_test.append(tmp)

random.shuffle(data)
max_data_trainning = round(len(data) * 0.8)

for idex, value in enumerate(data) :
    has_key = 0
    for i in value['x']:
        if i == 1 :
            has_key += 1
    if has_key >= MIN_KEY_TRAIN:
        x_train.append(value['x'])
        y_train.append(value['y'])
    # if idex < max_data_trainning :
    #     x_train.append(value['x'])
    #     y_train.append(value['y'])
    # else :
    #     x_test.append(value['x'])
    #     y_test.append(value['y'])

print(len(x_train), len(x_test), len(y_train), len(y_test), len(keynum), len(skillnum) )

x = tf.placeholder(tf.float32, [None, len(keynum)])
y = tf.placeholder(tf.float32, [None, len(skillnum)])

z = tf.Variable(tf.zeros([len(keynum), len(skillnum)]))
b = tf.Variable(tf.zeros([len(skillnum)]))

y_ = tf.nn.sigmoid( tf.matmul(x, z)  + b )

loss = tf.reduce_sum(tf.square(y - y_))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

train = tf.train.GradientDescentOptimizer(GRADIAN_OPTIMIZER).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

writer = tf.summary.FileWriter("output", sess.graph)

for i in range(NUM_TRAIN):
    sess.run(train, {x:x_train, y:y_train})

writer.close()
# Test trained model
y_output = tf.round(y_)
correct_prediction = tf.equal( tf.round(y_), y)
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

y_output = sess.run( y_output, {x:x_test, y:y_test})
# print("accuracy", sess.run( accuracy, {x:x_test, y:y_test}))
# print(y_output)
# print(y_test)

True_Positive = 0
False_Positive = 0
True_Negative = 0
False_Negative = 0

for idex, value in enumerate(y_output) :
    for jdex, val in enumerate(value) :
        if val == 1 and y_test[idex][jdex] == 1:
            True_Positive += 1
        if val == 1 and y_test[idex][jdex] == 0:
            False_Positive += 1
        if val == 0 and y_test[idex][jdex] == 0:
            True_Negative += 1
        if val == 0 and y_test[idex][jdex] == 1:
            False_Negative += 1

Precision = True_Positive / ( True_Positive + False_Positive )
TPR = True_Positive / ( True_Positive + False_Negative )
print("Precision", Precision)
print("TPR", TPR)

# ---------------------


# print(model.wv.similar_by_word('for'))
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=3)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()
