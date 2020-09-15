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
clsf_l = svm(kernel='linear') 
clsf_e = svm(kernel='gaus',non_linear_parametro=0.1)
#polinomial não ta funcionando. (não sei o pq)
#clsf_p = svm(kernel='poli',non_linear_parametro=3) 

#teste
clsf_t_rbf = SVC(kernel='rbf')
clsf_t_l = SVC(kernel='linear')
clsf_t_poli = SVC(kernel='poly')


clsf_l.fit(x_trein,y_trein)
clsf_e.fit(x_trein,y_trein)
#clsf_p.fit(x_trein,y_trein)

clsf_t_rbf.fit(x_trein,y_trein)
clsf_t_l.fit(x_trein,y_trein)
clsf_t_poli.fit(x_trein,y_trein)

yi_l = [clsf_l.predict(x) for x in x_test]
yi_e = [clsf_e.predict(x) for x in x_test]
#yi_p = [clsf_p.predict(x) for x in x_test]

yi_t_rbf = [clsf_t_rbf.predict([x]) for x in x_test]
yi_t_l = [clsf_t_l.predict([x]) for x in x_test]
yi_t_poli = [clsf_t_poli.predict([x]) for x in x_test]


#print('\n\n\n',yi_l,'\n\n\n',y_test)

print('Acuracia:')
print('linear: ',form.accuracy(y_test,yi_l))
print('Gaussiano: ',form.accuracy(y_test,yi_e))
#print('Polinomial: ',form.accuracy(y_test,yi_p))
print('(test)SVC rfb: ',form.accuracy(y_test,yi_t_rbf))
print('(test)SVC linear: ',form.accuracy(y_test,yi_t_l))
print('(test)SVC poly: ',form.accuracy(y_test,yi_t_poli))


