import time
import datetime
import csv
import pickle

import operator

print("Started data preprocessing at " + str(datetime.datetime.now()) + " ...")

# Load .csv dataset
with open("data_raw/yoochoose-data/yoochoose-clicks.dat", "rb") as f:
    reader = csv.DictReader(f, delimiter=',', fieldnames=['SessionId', 'Time', 'ItemId', 'Category'])
    sess_clicks_train = {}
    sess_date_train = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['SessionId']
        if curdate and not curid == sessid:
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date_train[curid] = date
        curid = sessid
        item = data['ItemId']
        curdate = data['Time'].split("T")[0]
        if sess_clicks_train.has_key(sessid):
            sess_clicks_train[sessid] += [item]
        else:
            sess_clicks_train[sessid] = [item]
        ctr += 1
        if ctr % 100000 == 0:
            print ('Loaded', ctr)
        # if ctr > 1000000:
        #     break
    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    sess_date_train[curid] = date

with open("data_raw/yoochoose-data/yoochoose-test.dat", "rb") as f:
    reader = csv.DictReader(f, delimiter=',', fieldnames=['SessionId', 'Time', 'ItemId', 'Category'])
    sess_clicks_test = {}
    sess_date_test = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['SessionId']
        if curdate and not curid == sessid:
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date_test[curid] = date
        curid = sessid
        item = data['ItemId']
        curdate = data['Time'].split("T")[0]
        if sess_clicks_test.has_key(sessid):
            sess_clicks_test[sessid] += [item]
        else:
            sess_clicks_test[sessid] = [item]
        ctr += 1
        if ctr % 100000 == 0:
            print ('Loaded', ctr)
        # if ctr > 1000000:
        #     break
    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
    sess_date_test[curid] = date

print(len(sess_clicks_train))
print(len(sess_date_train))

print(len(sess_clicks_test))
print(len(sess_date_test))

# Filter out length 1 sessions
for s in sess_clicks_train.keys():
    if len(sess_clicks_train[s]) == 1:
        del sess_clicks_train[s]
        del sess_date_train[s]

for s in sess_clicks_test.keys():
    if len(sess_clicks_test[s]) == 1:
        del sess_clicks_test[s]
        del sess_date_test[s]

# Merge
sess_clicks = sess_clicks_train.copy()
sess_clicks.update(sess_clicks_test)

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid_counts.has_key(iid):
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

for s in sess_clicks_train.keys():
    curseq = sess_clicks_train[s]
    filseq = filter(lambda i: iid_counts[i] >= 5, curseq)
    if len(filseq) < 2:
        del sess_clicks_train[s]
        del sess_date_train[s]
    else:
        sess_clicks_train[s] = filseq

for s in sess_clicks_test.keys():
    curseq = sess_clicks_test[s]
    filseq = filter(lambda i: iid_counts[i] >= 5, curseq)
    if len(filseq) < 2:
        del sess_clicks_test[s]
        del sess_date_test[s]
    else:
        sess_clicks_test[s] = filseq

train_sess = sess_date_train.items()
test_sess = sess_date_test.items()

# Sort sessions by date
train_sess = sorted(train_sess, key=operator.itemgetter(1))
test_sess = sorted(test_sess, key=operator.itemgetter(1))

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}
item_ctr = 1
train_seqs = []
train_dates = []
my_train_ctr = 0
# Convert training sessions to sequences and renumber items to start from 1
for s, date in train_sess:
    seq = sess_clicks_train[s]
    outseq = []
    for i in seq:
        if item_dict.has_key(i):
            outseq += [item_dict[i]]
        else:
            outseq += [item_ctr]
            item_dict[i] = item_ctr
            item_ctr += 1
    if len(outseq) < 2:  # Doesn't occur
        continue
    my_train_ctr += 1
    train_seqs += [outseq]
    train_dates += [date]

print("Number of training sessions:")
print(my_train_ctr)

test_seqs = []
test_dates = []
my_test_ctr = 0
# Convert test sessions to sequences, ignoring items that do not appear in training set
for s, date in test_sess:
    seq = sess_clicks_test[s]
    outseq = []
    for i in seq:
        if item_dict.has_key(i):
            outseq += [item_dict[i]]
    if len(outseq) < 2:
        continue
    my_test_ctr += 1
    test_seqs += [outseq]
    test_dates += [date]

print("Number of test sessions:")
print(my_test_ctr)

print("Number of items:")
print(item_ctr)

def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    for seq, date in zip(iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]

    return out_seqs, out_dates, labs

tr_seqs, tr_dates, tr_labs = process_seqs(train_seqs,train_dates)
te_seqs, te_dates, te_labs = process_seqs(test_seqs,test_dates)

train = (tr_seqs, tr_labs)
test = (te_seqs, te_labs)

print("Number of training examples (sequences):")
print(len(tr_labs))
print("Number of test examples (sequences):")
print(len(te_labs))

print("Finished data preprocessing at " + str(datetime.datetime.now()) + " ...")

print("Started writing to file at " + str(datetime.datetime.now()) + " ...")

f1 = open('data/yc_train_dummy.pkl', 'w')
pickle.dump(train, f1)
f1.close()
f2 = open('data/yc_test_dummy.pkl', 'w')
pickle.dump(test, f2)
f2.close()

print("Finished writing to file at " + str(datetime.datetime.now()) + " ...")
