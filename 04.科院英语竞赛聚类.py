from sklearn.cluster import KMeans
import numpy as np

colleges = []
scores = []
counts = []
f = open('英语竞赛.csv')

for line in f:
    gender = line.split(',')[2] #性别
    college = line.split(',')[5] #学院
    score = float(line.split(',')[6])
    if college not in colleges:
        colleges.append(college)
        scores.append(score)
        counts.append(1)
    else:
        scores[colleges.index(college)] += score
        counts[colleges.index(college)] += 1

print(colleges)
print(scores)
print(counts)
average = []
for i in range(0,len(colleges)):
    average.append([float(scores[i] / counts[i])])
print(average)

km = KMeans(n_clusters=5)

label = km.fit_predict(average)
Average = np.sum(km.cluster_centers_,axis=1)
print(Average)
College_result = [[],[],[],[],[]]
for i in range(len(colleges)):
    College_result[label[i]].append(colleges[i])
for i in range(len(College_result)):
    print("average:%.2f"%Average[i])
    print(College_result[i])