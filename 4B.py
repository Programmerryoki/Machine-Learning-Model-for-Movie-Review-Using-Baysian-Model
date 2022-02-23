from os import listdir
from sklearn.model_selection import train_test_split
from json import dump
from math import log

def clean_text(txt):
    txt = txt.strip()
    txt = txt.lower()
    return txt

original_data = {}
while True:
    path = input("Please enter the path to folder which has both positive and negative movie reviews!\n")
    try:
        for folder in listdir(path):
            original_data[folder] = []
            for file_name in listdir("/".join([path, folder])):
                with open("/".join([path,folder,file_name]), "r") as file:
                    original_data[folder].append(file.readlines())
    except:
        print("Error occured while trying to read files, please reenter the path!")
        continue
    finally:
        if original_data:
            print("Reading Succeeded")
            break

txt_data = []
for folder in original_data:
    for review in original_data[folder]:
        txt_data.append(("".join(review), folder))

raw_train_set, raw_test_set = train_test_split(txt_data, train_size=0.9, random_state=2021)

bayesian = {}
for raw_data, label in raw_train_set:
    for raw_word in raw_data.split():
        word = clean_text(raw_word)
        if word in bayesian:
            if label not in bayesian[word]:
                bayesian[word][label] = 0
            bayesian[word][label] += 1
        else:
            bayesian[word] = {}
            bayesian[word][label] = 1

print("Writing data to baysian_data.txt!")
with open("Bayesian_data.txt", "w") as file:
    dump(bayesian, file)

count = {}
for raw_data,label in raw_train_set:
    if label not in count:
        count[label] = 0
    count[label] += 1
evidence_level = []
labels = list(sorted(count.keys()))
for word in bayesian:
    tmp = []
    for label in labels:
        if label not in bayesian[word]:
            tmp.append(-float("inf"))
        else:
            denominator = sum(bayesian[word].values()) * count[label]
            numerator = bayesian[word][label] * sum(count.values())
            p = log(numerator / denominator, 2)
            tmp.append(p)
    evidence_level.append((word, tuple(tmp)))

print("Writing top5 evidences to top5evidence.txt!")
with open("top5evidence.txt", "w") as file:
    for i in range(len(labels)):
        file.write(f"Top 5 {labels[i]} words\n")
        evidence_level.sort(key=lambda x: (x[1][i], x[1][1-i]))
        for j in range(5):
            file.write(f"{j+1} {evidence_level[j][0]}\n")