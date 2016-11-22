import Network
import csv_loader
import numpy as np
import csv

training_input, training_output, test_data,test_ds = csv_loader.load_csv_files()
training_data = zip(training_input,training_output)


net = Network.Network([784, 40, 10])

net.SGD(training_data, 40, 10, 3.0)

test_results = [np.argmax(net.feedforward(x)) for x in test_data]


with open('names.csv', 'w') as csvfile:
    fieldnames = ['ImageId', 'Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i,results in enumerate(test_results):
        writer.writerow({'ImageId': i+1, 'Label': results})


print len(test_results)