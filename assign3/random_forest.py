
import random
import pprint
from sklearn.ensemble import RandomForestClassifier

def RandomForest(X, Y, xt, yt, nE, mD):
    clf = RandomForestClassifier(n_estimators = nE ,max_depth=mD, verbose=0, random_state=0, oob_score=True)
    clf.fit(X, Y)
    
    predl = clf.predict(xt)
    
    return clf.oob_score_, predl, clf.estimators_, clf.classes_

def findOptimum(X, Y, xt, yt, mnE, mmD):
    onE = -1
    omD = -1
    err = -1
    for i in xrange(1, mnE + 1):
        for j in xrange(1, mmD + 1):
            newerr, _, _, _ = RandomForest(X, Y, xt, yt, i, j)
            if(newerr > err):
                onE = i
                omD = j
                err = newerr
    return onE, omD

def compareEstimators(xt, yt, ests, classes):
    print("Error in each estimator")
    for estimator in ests:
        predl = estimator.predict(xt)
        err = 0
        for i in range(len(predl)):
            if(classes[int(predl[i])] != yt[i]):
                err += 1
        print ((err * 100.0) / len(predl))

def main():
    f1 = open("breastcancer.txt")
    
    dataset = []
    for line in f1:
        values = line.rstrip().split(',')
        if ("?" in values):
            continue
        elif len(values) > 1:
            dataset.append(values)
    
    random.shuffle(dataset)
    
    sixty = int(len(dataset) * 0.6)
    
    train = dataset[:sixty]
    test = dataset[sixty:]    
    
    X = [lst[1:len(lst) - 1] for lst in train]
    Y = [lst[len(lst) - 1] for lst in train]
    
    xt = [lst[1:len(lst) - 1] for lst in test]
    yt = [lst[len(lst) - 1] for lst in test]
    
    optE, optD = findOptimum(X, Y, xt, yt, 50, 50)
    
    print "Optimum values of number of estimators and maximum depth: ", (optE, optD)
    
    err, labels, ests, classes = RandomForest(X, Y, xt, yt, optE, optD)
    print "Error: ", err
    
    compareEstimators(xt, yt, ests, classes)
    
    
if __name__ == '__main__':
    main()