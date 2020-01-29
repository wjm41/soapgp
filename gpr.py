from typing import Tuple, Optional
import time 
import numpy as np
import sys
import pandas as pd
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from helper import scaffold_split
from scipy.stats import spearmanr as spear

data_name = sys.argv[1]
method_name = sys.argv[2]

if data_name=='Malaria':
	ran_state_list=[6]
else:
	ran_state_list = range(4,7)
rem_mat = np.load('/rds-d2/user/wjm41/hpc-work/kernels/dscribe/'+data_name+'_'+method_name+'_kernel_rematch.npy')

print('Analysing '+data_name+' dataset with SOAP-GP...')

csv_name = 'dmap_minimal/data/'+data_name+'.csv'

if data_name=='lipo':
	csv_headers=['Name','lip_val','SMILES']
	pred_val = 'lip_val'
elif data_name=='esol':	
	csv_headers=['Name','pred_ESOL','min_deg','mol_weight','num_H_donor','num_rings','num_rot_bonds','surface_area','expt_ESOL','SMILES']
	pred_val = 'expt_ESOL'
elif data_name=='FreeSolv':
	csv_headers=['Name','SMILES','expt','calc']
	pred_val = 'expt'
elif data_name=='CatS':
	csv_headers=['SMILES','val','Name']
	pred_val = 'val'
elif data_name=='nusolv':
	csv_headers=['Chemical name','Crystal structure CSD refcode','LogS exp (mol/L)','Reference','Test','SMILES','Smiles Source','Chemspider reference number','InChl']
	pred_val = 'LogS exp (mol/L)'
elif data_name=='bradley':
	csv_headers=["SMILES","CASRN","EXTERNALID","N","NAME","NAME","ARTICLEID","PUBMEDID","PAGE","TABLE","Melting Point {measured}","UNIT {Melting Point}","Melting Point {measured, converted}","UNIT {Melting Point}"]
	pred_val = 'Melting Point {measured, converted}'
elif data_name=='Malaria':
	csv_headers=['EXT_CMPD_NUMBER','SMILES','Percentage_inhibition_3D7','Percentage_inhibition_DD2','Percentage_inhibition_3D7_PFLDH','XC50_MOD_3D7','XC50_3D7 (microM)','Percentage_inhibition_HEPG2']
#	pred_vals = ['Percentage_inhibition_3D7','Percentage_inhibition_DD2','Percentage_inhibition_3D7_PFLDH','XC50_3D7 (microM)','Percentage_inhibition_HEPG2']
	pred_val = 'XC50_3D7 (microM)'
else:
   print('Unrecognised data file')

len_dataset = len(rem_mat[0])

#Split into features - dataset specific

#smiles_list = []
rem_mat=np.float64(rem_mat)
csv = pd.read_csv(csv_name, header=None,names=csv_headers)

X = np.where(((csv[pred_val]!='ND') & (csv[pred_val]!='<')) & (csv[pred_val].notnull()))[0]
X=X.reshape(-1,1)
filtered_csv = csv[((csv[pred_val]!='ND') & (csv[pred_val]!='<')) & (csv[pred_val].notnull())]

Y = filtered_csv[pred_val].astype('float').values
if data_name=='Malaria':
	Y = np.log10(Y)
Y=Y.reshape(-1,1)

scaler=StandardScaler(with_std=False)

smiles_list = filtered_csv['SMILES'].values

max_rem = rem_mat.max()
rem_diag = tf.constant(np.diag(rem_mat),dtype=tf.float64)
rem_mat = tf.constant(rem_mat,dtype=tf.float64)

def rem_sub(A,A2):
	global rem_mat
	A = tf.cast(A,tf.int32)
	A = tf.reshape(A,[-1])
	A2 = tf.reshape(A2,[-1])
	A2 = tf.cast(A2,tf.int32)
	K_mat = tf.gather(rem_mat, A, axis=0)
	K_mat = tf.gather(K_mat, A2, axis=1)
	return tf.cast(K_mat, tf.float64)

from gpflow.utilities import positive
class Matern32_rem(gpflow.kernels.Kernel):
	global rem_mat, rem_diag, max_rem
	def __init__(self):
		super().__init__(active_dims=[0])
		self.var = gpflow.Parameter(1.0, transform=positive())
		self.mag = gpflow.Parameter(1.0, transform=positive())
	
	def K(self, A, A2=None, presliced=None):
		if A2 is None:
			A2=A
		K_mat = rem_sub(A,A2)
		z = tf.math.sqrt(6*(max_rem-K_mat))*self.var
		K_final = self.mag*(1+z)*tf.math.exp(-z)
		return K_final
	def K_diag(self, A, presliced=None):
		A=tf.cast(A,tf.int32)
		K_diag = tf.cast(tf.gather_nd(rem_diag, A),tf.float64)
		z = tf.math.sqrt(6*(max_rem-K_diag))*self.var
		return self.mag*(1+z)*tf.math.exp(-z)


#Data_set split
from sklearn.model_selection import KFold, train_test_split
r2_list = []
rmse_list = []
mae_list = []
spear_list = []

model=None

def objective_closure():
	mar_lik  = -model.log_marginal_likelihood()
	return mar_lik

counter=0
for i in ran_state_list:
	kf = KFold(n_splits=5, random_state=i, shuffle=True)
	j=0
	for train_ind, test_ind in kf.split(X):
		#train_ind, test_ind = scaffold_split(smiles_list, balanced=True, seed=i)
		X_train, X_test = X[train_ind], X[test_ind]
		y_train, y_test = Y[train_ind], Y[test_ind]
		_ , X_val, _ , y_val = train_test_split(X_train, y_train, train_size=0.857, random_state=i)

		y_train = scaler.fit_transform(y_train)
		y_val = scaler.transform(y_val)
		y_test = scaler.transform(y_test)       

		X_train = tf.convert_to_tensor(X_train, dtype = tf.float64)
		X_test = tf.convert_to_tensor(X_test, dtype = tf.float64)
		if counter==0:
			model = gpflow.models.GPR(data=(X_train,y_train),kernel= Matern32_rem() + gpflow.kernels.White(0.1))
		else: 
			model.X = X_train
			model.Y = y_train
			model.kernel = Matern32_rem() + gpflow.kernels.White(0.1)
		opt = gpflow.optimizers.Scipy()

		opt_logs = opt.minimize(objective_closure, model.trainable_variables, options=dict(maxiter=10000))
		#mean and variance GP prediction
		y_pred, y_var = model.predict_y(X_test)
		y_pred_train, _ = model.predict_y(X_train)
		#print(y_test)
		print('R2 on training set: '+str(r2_score(y_train, y_pred_train)))	
	
		score = r2_score(y_test, y_pred)

		spear_score = spear(np.absolute(y_test-y_pred),y_var)[0]
		rmse = np.sqrt(mean_squared_error(y_test, y_pred))
		mae = mean_absolute_error(y_test, y_pred) 
		print("\nR^2: {:.3f}".format(score))
		print("RMSE: {:.3f}".format(rmse))
		print("MAE: {:.3f}".format(mae))
		print("rho: {:.3f}".format(spear_score))
		r2_list.append(score)
		rmse_list.append(rmse)
		mae_list.append(mae)
		spear_list.append(spear_score)
		#y_pred = scaler.inverse_transform(y_pred)
		#np.savetxt(data_name+'_ytest.dat',y_test)
		#np.savetxt('combo_data/'+data_name+'_scaff_ypred_seed_'+str(i)+'_soap.dat',y_pred)
		#np.savetxt('combo_data/'+data_name+'_scaff_ytest_seed_'+str(i)+'.dat',y_test)
		#np.savetxt('combo_data/'+data_name+'_scaff_yvar_seed_'+str(i)+'_soap.dat',np.sqrt(y_var))
		#np.savetxt('combo_data/'+data_name+'_ypred_seed_'+str(i)+'_fold_'+str(j)+'_soap.dat',y_pred)
		#np.savetxt('combo_data/'+data_name+'_ytest_seed_'+str(i)+'_fold_'+str(j)+'.dat',y_test)
		#np.savetxt('combo_data/'+data_name+'_yvar_seed_'+str(i)+'_fold_'+str(j)+'_soap.dat',np.sqrt(y_var))
		j+=1
r2_list = np.array(r2_list)
rmse_list = np.array(rmse_list)
mae_list = np.array(mae_list)
spear_list = np.array(spear_list)
print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list),np.std(rmse_list)/np.sqrt(len(rmse_list))))
print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list),np.std(mae_list)/np.sqrt(len(mae_list))))
print("mean rho between predicted error and actual error: {:.4f} +- {:.4f}\n".format(np.mean(spear_list),np.std(spear_list)/np.sqrt(len(spear_list))))
