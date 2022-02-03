from tensorflow import keras
from keras import datasets
#from tensorflow.python.keras impordatasets
import numpy as np
from numpy import array
from numpy.linalg import norm
import pickle

import IISL_FLpkg.model_generator_non_iid as mg

N = 100
L = 3
L2 = 10
p = 0.4

sca_metric = keras.metrics.SparseCategoricalAccuracy(name="sca")
p_sca_metric = keras.metrics.SparseCategoricalAccuracy(name="p_sca")
rp_sca_metric = keras.metrics.SparseCategoricalAccuracy(name="rp_sca")
rp2_sca_metric = keras.metrics.SparseCategoricalAccuracy(name="rp2_sca")


all_models, central_server = mg.model_generation(N, sca_metric)
p_all_models, p_central_server = mg.model_generation(N, p_sca_metric)
rp_all_models, rp_central_server = mg.model_generation(N, rp_sca_metric)
rp2_all_models, rp2_central_server = mg.model_generation(N, rp2_sca_metric)

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
x_train, x_test = x_train / 255.0, x_test / 255.0
x_test = x_test[0:100]
y_test = y_test[0:100]
x_train_copy = x_train.copy()
y_train_copy = y_train.copy()

x_train_iid = x_train_copy[0:30000]
y_train_iid = y_train_copy[0:30000]
x_train_niid = x_train_copy[30000:60000]
y_train_niid = y_train_copy[30000:60000]

for index, val in enumerate(y_train_niid):
    if val > 4 and index > 0:
        y_train_niid[index] = y_train_niid[index - 1]
        x_train_niid[index] = x_train_niid[index - 1]


loss_list = []
p_loss_list = []
rp_loss_list = []
rp2_loss_list = []

accuracy_list = []
p_accuracy_list = []
rp_accuracy_list = []
rp2_accuracy_list = []


for iter in range(4):
  for i in range(600):
    x = np.concatenate([x_train_iid[50*(i):50*(i+1)], x_train_niid[50*(i):50*(i+1)]])
    y = np.concatenate([y_train_iid[50*(i):50*(i+1)], y_train_niid[50*(i):50*(i+1)]])

    results = all_models.fed_avg_non_iid(x, y, x_test, y_test, sca_metric, central_server)
    loss_list.append(results[0])
    accuracy_list.append(results[1])

    p_results = p_all_models.pfed_avg_non_iid(x, y, x_test, y_test, p_sca_metric, p_central_server, L, i)
    p_loss_list.append(p_results[0])
    p_accuracy_list.append(p_results[1])

    rp_results = rp_all_models.rpfed_avg_non_iid(x, y, x_test, y_test, rp_sca_metric, rp_central_server, 0.4, L, i)
    rp_loss_list.append(rp_results[0])
    rp_accuracy_list.append(rp_results[1])
    
    rp2_results = rp2_all_models.rpfed_avg_non_iid(x, y, x_test, y_test, rp2_sca_metric, rp2_central_server, 0.1, L, i)
    rp2_loss_list.append(rp2_results[0])
    rp2_accuracy_list.append(rp2_results[1])
    
    # r_results = r_all_models.rfed_avg(x, y, r_central_server, 0.1)
    # r_loss_list.append(r_results[0])
    # r_accuracy_list.append(r_results[1])

    # rq_results = rq_all_models.rqfed_avg(x, y, rq_central_server, 0.1)
    # rq_loss_list.append(rq_results[0])
    # rq_accuracy_list.append(rq_results[1])

    if(i % 100 == 0):
      print("iteration : ", iter, ", i : ", i)
      print("loss : %.7f, sca : %.7f" %( results[0], results[1]))
      print("[P]loss : %.7f, sca : %.7f" %( p_results[0], p_results[1]))
      print("[RP]loss : %.7f, sca : %.7f" %( rp_results[0], rp_results[1]))
      print("[RP2]loss : %.7f, sca : %.7f" %( rp2_results[0], rp2_results[1]))
    #   print("[R]loss : %.7f, sca : %.7f" %( r_results[0], r_results[1]))
    #   print("[RQ]loss : %.7f, sca : %.7f" %( rq_results[0], rq_results[1]))
    
with open("./Accuracy_lists/OFedAvg(non).pkl","wb") as f:
    pickle.dump(accuracy_list, f)
    
with open("./Accuracy_lists/OFedIT(non)_L3_p1.0.pkl","wb") as f:
    pickle.dump(p_accuracy_list, f)
    
with open("./Accuracy_lists/OFedIT(non)_L3_p0.4.pkl","wb") as f:
    pickle.dump(rp_accuracy_list, f)
    
with open("./Accuracy_lists/OFedIT(non)_L3_p0.1.pkl","wb") as f:
    pickle.dump(rp2_accuracy_list, f)