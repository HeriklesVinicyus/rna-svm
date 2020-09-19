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
clsf_laphace_gaussian = svm(kernel='lrbf',non_linear_parametro=4.0)
clsf_hyperbolic_tanh = svm(kernel='tanh', a=(x_trein.shape[1]/2), const=3)
clsf_exponecial = svm(kernel='exp',non_linear_parametro=4.0)

#clsf_p = svm(kernel='poli',non_linear_parametro=2)

#teste
clsf_t_rbf = SVC(kernel='rbf')
clsf_t_l = SVC(kernel='linear',C=1000)
clsf_t_poli = SVC(kernel='poly')

clsf_linear.fit(np.array(x_trein),np.array(y_trein))
clsf_gaussian.fit(np.array(x_trein),np.array(y_trein))
#clsf_p.fit(np.array(x_trein),np.array(y_trein))#erro
clsf_laphace_gaussian.fit(np.array(x_trein),np.array(y_trein))
clsf_hyperbolic_tanh.fit(np.array(x_trein),np.array(y_trein))
clsf_exponecial.fit(np.array(x_trein),np.array(y_trein))

clsf_t_rbf.fit(x_trein,y_trein)
clsf_t_l.fit(x_trein,y_trein)
clsf_t_poli.fit(x_trein,y_trein)

yi_l = [clsf_linear.predict(x) for x in x_test]
yi_g = [clsf_gaussian.predict(x) for x in x_test]
#yi_p = [clsf_p.predict(x) for x in x_test]
yi_lg = [clsf_laphace_gaussian.predict(x) for x in x_test]
yi_th = [clsf_hyperbolic_tanh.predict(x) for x in x_test]
yi_exp = [clsf_exponecial.predict(x) for x in x_test]


yi_t_rbf = [clsf_t_rbf.predict([x]) for x in x_test]
yi_t_l = [clsf_t_l.predict([x]) for x in x_test]
yi_t_poli = [clsf_t_poli.predict([x]) for x in x_test]


print('Acuracia:')
print('linear: ',form.accuracy(y_test,yi_l))
print('Gaussiano: ',form.accuracy(y_test,yi_g))
#print('Polinomial: ',form.accuracy(y_test,yi_p))
print('lrbf: ',form.accuracy(y_test,yi_lg))
print('Tangente Hiper: ',form.accuracy(y_test,yi_th))
print('Exponencial: ',form.accuracy(y_test,yi_exp))

print('\n(test)SVC rfb: ',form.accuracy(y_test,yi_t_rbf))
print('(test)SVC linear: ',form.accuracy(y_test,yi_t_l))
print('(test)SVC poly: ',form.accuracy(y_test,yi_t_poli))