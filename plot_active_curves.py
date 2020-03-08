"""
Plots active learning experiment results (active learning vs random sampling) using data from /results/
"""
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

method=sys.argv[1]
task=sys.argv[2]
n_runs = int(sys.argv[3])

scaff_std = np.load('results/'+method+'_'+task+'_samp_var_split_scaffold_seed_0.npy')
scaff_std = scaff_std.reshape(-1,1)
scaff_rand = np.load('results/'+method+'_'+task+'_samp_rand_split_scaffold_seed_0.npy')
scaff_rand = scaff_rand.reshape(-1,1)
#rand_std = np.load('results/'+method+'_'+task+'_samp_var_split_random_seed_0.npy')
#rand_std = rand_std.reshape(-1,1)
#rand_rand = np.load('results/'+method+'_'+task+'_samp_rand_split_random_seed_0.npy')
#rand_rand = rand_rand.reshape(-1,1)

for i in range(n_runs-1):
	scaff_std = np.hstack((scaff_std, np.load('results/'+method+'_'+task+'_samp_var_split_scaffold_seed_'+str(i)+'.npy').reshape(-1,1)))
	#rand_std = np.hstack((rand_std, np.load('results/'+method+'_'+task+'_samp_var_split_random_seed_'+str(i)+'.npy').reshape(-1,1)))
	scaff_rand = np.hstack((scaff_rand, np.load('results/'+method+'_'+task+'_samp_rand_split_scaffold_seed_'+str(i)+'.npy').reshape(-1,1)))
	#rand_rand = np.hstack((rand_rand, np.load('results/'+method+'_'+task+'_samp_rand_split_random_seed_'+str(i)+'.npy').reshape(-1,1)))

scaff_std_mean = np.mean(scaff_std, axis=1)
scaff_rand_mean = np.mean(scaff_rand, axis=1)
#rand_std_mean = np.mean(rand_std, axis=1)
#rand_rand_mean = np.mean(rand_rand, axis=1)

scaff_std_std = np.std(scaff_std, axis=1)/np.sqrt(len(scaff_std))
scaff_rand_std = np.std(scaff_rand, axis=1)/np.sqrt(len(scaff_std))
#rand_std_std = np.std(rand_std, axis=1)/np.sqrt(len(rand_std))
#rand_rand_std = np.std(rand_rand, axis=1)/np.sqrt(len(rand_std))

plt.clf()
x = np.arange(11)
plt.title(method+'-GP on '+task+' '+str(n_runs)+' runs')
plt.plot(x, scaff_std_mean, 'C0',label='scaffold split active')
plt.fill_between(x,\
                 scaff_std_mean - scaff_std_std,\
                 scaff_std_mean + scaff_std_std,\
                 color='C0', alpha=0.2)
plt.plot(x, scaff_rand_mean, linestyle='--', color='C3',label='scaffold split random')
plt.fill_between(x,\
                 scaff_rand_mean - scaff_rand_std,\
                 scaff_rand_mean + scaff_rand_std,\
                 color='C3', alpha=0.2)
#plt.plot(x, rand_std_mean, 'C1',label='random split active')
#plt.fill_between(x,\
#                 rand_std_mean - rand_std_std,\
#                 rand_std_mean + rand_std_std,\
#                 color='C1', alpha=0.2)
#plt.plot(x, rand_rand_mean, linestyle='--', color='C2',label='random split random')
#plt.fill_between(x,\
#                 rand_rand_mean - rand_rand_std,\
#                 rand_rand_mean + rand_rand_std,\
#                 color='C2', alpha=0.2)
plt.xlabel('Number of iterations')
plt.ylabel('RMSE')
plt.legend(loc='upper right')
plt.savefig(task+'_active_'+method+'.pdf')
