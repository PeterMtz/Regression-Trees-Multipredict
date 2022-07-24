# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

#%% Data generation
rng = np.random.RandomState(0)

# #############################################################################
# Generate sample data
X = np.zeros((100,3))
X[:,0] = 5 * rng.rand(100, 1)[:,0]
X[:,1] = 10 * rng.rand(100, 1)[:,0]
X[:,2] = 15 * rng.rand(100, 1)[:,0]

y = np.ravel(3*X[:,0]-2*X[:,1]+3*X[:,2]-10)
# Add noise to targets
yrnd = y + 3 * (0.5 - rng.rand(y.shape[0]))


plt.figure(figsize=(12,4))
plt.subplot(131)
plt.scatter(X[:,0], yrnd, c='r', s=10, label='Conjunto S',zorder=2)
plt.xlabel('x_p1')
plt.ylabel('y')
plt.legend()
plt.grid()

plt.subplot(132)
plt.scatter(X[:,1], yrnd, c='r', s=10, label='Conjunto S',zorder=2)
plt.xlabel('x_p2')
plt.ylabel('y')
plt.legend()
plt.grid()

plt.subplot(133)
plt.scatter(X[:,2], yrnd, c='r', s=10, label='Conjunto S',zorder=2)
plt.xlabel('x_p3')
plt.ylabel('y')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


#%% Build and fit a decision tree regressor
model = DecisionTreeRegressor(random_state=0,
                               splitter='best',
                               max_depth=None,
                               min_samples_split=2,
                               min_samples_leaf=1)
model = model.fit(X,yrnd)
Yhat = model.predict(X)
print('R2 = %0.4f'%model.score(X,yrnd))

plt.figure(figsize=(8,8))
plt.scatter(yrnd,Yhat,c='r',label='datos-ruido')
plt.plot(np.linspace(-20,50,100),np.linspace(-20,50,100),'g--',label='ajuste')
plt.xlabel('y_real')
plt.ylabel('Y_estimada')
plt.grid()
plt.legend()
plt.show()

#%% View decision tree
from sklearn import tree
tree.plot_tree(model) 

#%% Export decision tree
import graphviz
dot_data = tree.export_graphviz(model, out_file=None)
graph = graphviz.Source(dot_data)
graph.render(filename='trees_gen/tree_multipredict',
             format='pdf')

#%% Find the best depth as hyperparameter
depths = np.arange(11)
models = []
model_scores = np.zeros(np.shape(depths))
for md in depths:
    model = DecisionTreeRegressor(random_state=0,
                               splitter='best',
                               max_depth=md+1,
                               min_samples_split=2,
                               min_samples_leaf=1)
    model = model.fit(X,yrnd)
    model_scores[md] = model.score(X,yrnd)
    models.append(model)

#%% View the score performance vs max_depth
grad_score = np.diff(model_scores)


plt.figure()
plt.plot(depths+1,model_scores)
plt.xlabel('Max_depths'),plt.ylabel('R^2 score')
plt.grid()

plt.figure()
plt.plot(depths[1:]+1,grad_score)
plt.xlabel('Max_depths'),plt.ylabel('diff score')
plt.grid()