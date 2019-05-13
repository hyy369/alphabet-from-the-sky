import csv
from sklearn.cluster import KMeans

with open('train_info.csv', 'r') as readFile:
    reader = csv.reader(readFile)
    lines = list(reader)
    for line in lines:
        line[0] = int(line[0])
        line[1] = int(line[1])
        line[2] = int(line[2])
readFile.close()

clt = KMeans(n_clusters=2) #cluster number
clt.fit(lines)
print(clt.labels_)
for i in range (len(lines)):
    lines[i].append(clt.labels_[i])
print(lines)

with open('after_train.csv', 'w') as csvFile:
    for line in lines:
        writer = csv.writer(csvFile)
        writer.writerow(line)
csvFile.close()