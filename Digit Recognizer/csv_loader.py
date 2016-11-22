import csv
import numpy as np
import random


def load_data(file_name,test_data):
    ifile = open(file_name, "rb")
    reader = csv.reader(ifile)
    rownum = 0
    training_input = []
    training_results = []

    for row in reader:
        if rownum == 0:
            header = row
        else:
            colnum = 0
            pix = []
            for col in row:
                if colnum == 0 and not test_data:
                    value = np.int64(col)
                else:
                    pix.append(np.float64(np.float64(col)/256))
                colnum += 1
            training_input.append(pix)
            if not test_data:
                training_results.append(value)

        rownum += 1

    ifile.close()
    if test_data:
        return training_input
    else:
        return training_input,training_results


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_csv_files():

    train_inp,train_res = load_data('train.csv',False)
    train_inp = [np.reshape(x, (784, 1)) for x in train_inp]
    test_ds =zip(train_inp,train_res)
    random.shuffle(test_ds)
    test_ds = test_ds[500:6500]
    train_res = [vectorized_result(y) for y in train_res]
    print len(train_inp)
    test_inp = load_data('test.csv',True)

    test_inp = [np.reshape(x, (784, 1)) for x in test_inp]
    print len(test_inp)
    for x in train_inp:
        print type(x)
        break

    for x in test_inp:
        print type(x)
        break

    return train_inp,train_res,test_inp,test_ds
