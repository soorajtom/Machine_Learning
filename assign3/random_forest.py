

def RandomForest():
    f1 = open("breastcancer.txt")
    
#    lines = f1.readlines()
    dataset = []
    for line in f1:
        p = []
        values = line.rstrip().split(',')
        if ("?" in values) or len(values) <= 1:
            continue
        else:
            for i in values:
                p.append(i)
        dataset.append(p)
    
    print dataset