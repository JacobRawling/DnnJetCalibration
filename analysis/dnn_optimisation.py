from ttbar_reconstructor_model import TTbarReconstructor
from bayes_opt import BayesianOptimization
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import cloudpickle
import numpy as np

train_columns = [ 'best_had_m_perm',
       'best_lep_m_perm',
       'W_lep_eta',
       'closest_to_W',
       'n_bjets',
       'furthest_from_W',
       'closest_to_met',
       'furthest_from_met',
       'closest_to_lep',
       'furthest_from_lep',
       'met_phi',
       'lep_pt',
       'best_reco_param'
       ]

train_columns = train_columns + ['dR_Lj%d'%i for i in range(5)]
train_columns = train_columns + ['dR_Wj%d'%i for i in range(5)]
train_columns = train_columns + ['jet_isBjet_%d'%i for i in range(5)]

ttbar_reconstructor = TTbarReconstructor(
    input_file_path='/hepgpu3-data1/jrawling/deep_tops/csvs_test/ttbar_[0-5].csv',
    input_columns=train_columns,
    epochs=30
    )


bo = BayesianOptimization(lambda x, y, z: ttbar_reconstructor(layers=[
                                int(x),int(y),int(z)
                            ]),
                          # define the target space
                          {
                            'x': (10, 300), 
                            'y': (10, 300), 
                            'z': (10, 300)  
                          }
                          )
# We explore the effect of the final layer 
print('Performing exploration...')
bo.explore({'x': [10, 10], 'y': [150, 150], 'z': [10, 50]})


# Using the probability of increase acquisition function
print('Performing maximisation...')
bo.maximize(init_points=0, n_iter=150, acq='poi', xi=3e2)

# Finally, we take a look at the final results.
print('Finished!')
print(bo.res)
# Full list of the results
print(bo.res['all'])

print(bo)

params = bo.res['all']
res = bo.res
points = []
for p in params['params']:
    points.append([int(p['x']), int(p['y']), int(p['z'])] )

f1_scores = res['all']['values']
plt.grid()
plt.scatter(np.array(points)[:,2], np.array(f1_scores),label='hidden layer 3',color='b')
plt.scatter(np.array(points)[:,1], np.array(f1_scores),label='hidden layer 2',color='r')
plt.scatter(np.array(points)[:,0], np.array(f1_scores),label='hidden layer 1',color='g')
plt.xlabel('Number of nodes')
plt.ylabel('F1 Score')
plt.ylim(0.0)
plt.legend(loc='best',edgecolor='black',frameon=True)
plt.savefig('f1_vs_nlayers.png')
