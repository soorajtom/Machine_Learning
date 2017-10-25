#Author : Sooraj Tom
#assignment 1 Question 1

fin = open('/home/labs/mac_ler/assign1/iris.data', "r")
fout = open('/home/labs/mac_ler/assign1/iris-svm-input.txt',"w")
lines = fin.readlines()
labels = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}

for line in lines:
    values = line.rstrip().split(',')
    if(len(values) >= 2):
        fout.write(str(labels[values[4]]))
        for i in range (4):
            if(float(values[i]) != 0):
                fout.write(" " + str(i + 1) + ":" + values[i])
        fout.write("\n")