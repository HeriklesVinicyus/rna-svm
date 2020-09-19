from classificador.SVM import svm
import classificador.form as form
import numpy as np
import matplotlib.pyplot as plt

#para teste 
from sklearn.svm import SVC


###Teste
database = []
with open('data_num.csv','r') as db_csv:
    read_data = db_csv.read()
    database = [x for x in read_data.split('\n')]
    database = [[float(i) for i in x.split(',')] for x in database]
db_csv.closed

X, y = form.split_Ys_from_data_base(form.shuffle_date_base(database),0)
X = np.array(X)
y = np.array(y)
y = np.where(y==2,-1,1)

x_trein, y_trein, x_test, y_test = form.split_in_training_and_test(X,y,80)

clsf_linear = svm(kernel='linear') 
clsf_gaussian = svm(kernel='gaus',non_linear_parametro=100)
clsf_laplace_gaussian = svm(kernel='lrbf',non_linear_parametro=25.0)
clsf_hyperbolic_tanh = svm(kernel='tanh', a=(x_trein.shape[1]/2), non_linear_parametro=3)
clsf_exponencial = svm(kernel='exp',non_linear_parametro=4.0)
clsf_qr = svm(kernel='qr',non_linear_parametro=7)
clsf_mq = svm(kernel='mq',non_linear_parametro=4)
clsf_mqi = svm(kernel='mqi',non_linear_parametro=5)
clsf_log = svm(kernel='log',non_linear_parametro=32)

#clsf_p = svm(kernel='poli',non_linear_parametro=2)


clsf_linear.fit(np.array(x_trein),np.array(y_trein))
clsf_gaussian.fit(np.array(x_trein),np.array(y_trein))
#clsf_p.fit(np.array(x_trein),np.array(y_trein))#erro
clsf_laplace_gaussian.fit(np.array(x_trein),np.array(y_trein))
clsf_hyperbolic_tanh.fit(np.array(x_trein),np.array(y_trein))
clsf_exponencial.fit(np.array(x_trein),np.array(y_trein))
#clsf_qr.fit(np.array(x_trein),np.array(y_trein))
#clsf_mq.fit(np.array(x_trein),np.array(y_trein))
clsf_mqi.fit(np.array(x_trein),np.array(y_trein))
#clsf_log.fit(np.array(x_trein),np.array(y_trein))


yi_l = [clsf_linear.predict(x) for x in x_test]
yi_g = [clsf_gaussian.predict(x) for x in x_test]
#yi_p = [clsf_p.predict(x) for x in x_test]
yi_lg = [clsf_laplace_gaussian.predict(x) for x in x_test]
yi_th = [clsf_hyperbolic_tanh.predict(x) for x in x_test]
yi_exp = [clsf_exponencial.predict(x) for x in x_test]
#yi_qr = [clsf_qr.predict(x) for x in x_test]
#yi_mq = [clsf_mq.predict(x) for x in x_test]
yi_mqi = [clsf_mqi.predict(x) for x in x_test]
#yi_log = [clsf_log.predict(x) for x in x_test]


#teste
clsf_t_rbf = SVC(kernel='rbf')
clsf_t_l = SVC(kernel='linear',C=1000)
clsf_t_poli = SVC(kernel='poly')

clsf_t_rbf.fit(x_trein,y_trein)
clsf_t_l.fit(x_trein,y_trein)
clsf_t_poli.fit(x_trein,y_trein)

yi_t_rbf = [clsf_t_rbf.predict([x]) for x in x_test]
yi_t_l = [clsf_t_l.predict([x]) for x in x_test]
yi_t_poli = [clsf_t_poli.predict([x]) for x in x_test]


print('Acuracia:')
print('linear: ',form.accuracy(y_test,yi_l))
print('Gaussiano: ',form.accuracy(y_test,yi_g))
#print('Polinomial: ',form.accuracy(y_test,yi_p))
print('lrbf: ',form.accuracy(y_test,yi_lg))
print('Tangente Hiperbólico : ',form.accuracy(y_test,yi_th))
print('Exponencial: ',form.accuracy(y_test,yi_exp))
#print('quadratico_racional : ',form.accuracy(y_test,yi_qr))
#print('Multiquadrico : ',form.accuracy(y_test,yi_mq))
print('multiquadrico inverso : ',form.accuracy(y_test,yi_mqi))
#print('log: ',form.accuracy(y_test,yi_log))


print('\n(test)SVC rfb: ',form.accuracy(y_test,yi_t_rbf))
print('(test)SVC linear: ',form.accuracy(y_test,yi_t_l))
print('(test)SVC poly: ',form.accuracy(y_test,yi_t_poli))