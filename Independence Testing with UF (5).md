```python
import proglearn
from proglearn.forest import UncertaintyForest
import hyppo
import numpy as np
from numba import njit
from hyppo.independence.base import IndependenceTest
from hyppo._utils import perm_test
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import copy 
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree._classes import DecisionTreeClassifier
from joblib import Parallel, delayed
from scipy.stats import entropy, multivariate_normal
from scipy.integrate import nquad
```

    Using TensorFlow backend.
    


```python
def uf(X, y, n_estimators = 300, max_samples = .4, base = np.exp(1), kappa = 3):
    
    # Build forest with default parameters.
    model = BaggingClassifier(DecisionTreeClassifier(), 
                              n_estimators=n_estimators, 
                              max_samples=max_samples, 
                              bootstrap=False)
    model.fit(X, y)
    n = X.shape[0]
    K = model.n_classes_
    _, y = np.unique(y, return_inverse=True)
    
    cond_entropy = 0
    for tree_idx, tree in enumerate(model):
        # Find the indices of the training set used for partition.
        sampled_indices = model.estimators_samples_[tree_idx]
        unsampled_indices = np.delete(np.arange(0,n), sampled_indices)
        
        # Randomly split the rest into voting and evaluation.
        total_unsampled = len(unsampled_indices)
        np.random.shuffle(unsampled_indices)
        vote_indices = unsampled_indices[:total_unsampled//2]
        eval_indices = unsampled_indices[total_unsampled//2:]
        
        # Store the posterior in a num_nodes-by-num_classes matrix.
        # Posteriors in non-leaf cells will be zero everywhere
        # and later changed to uniform.
        node_counts = tree.tree_.n_node_samples
        class_counts = np.zeros((len(node_counts), K))
        est_nodes = tree.apply(X[vote_indices])
        est_classes = y[vote_indices]
        for i in range(len(est_nodes)):
            class_counts[est_nodes[i], est_classes[i]] += 1
        
        row_sums = class_counts.sum(axis=1) # Total number of estimation points in each leaf.
        row_sums[row_sums == 0] = 1 # Avoid divide by zero.
        class_probs = class_counts / row_sums[:, None]
        
        # Make the nodes that have no estimation indices uniform.
        # This includes non-leaf nodes, but that will not affect the estimate.
        class_probs[np.argwhere(class_probs.sum(axis = 1) == 0)] = [1 / K]*K
        
        # Apply finite sample correction and renormalize.
        where_0 = np.argwhere(class_probs == 0)
        for elem in where_0:
            class_probs[elem[0], elem[1]] = 1 / (kappa*class_counts.sum(axis = 1)[elem[0]])
        row_sums = class_probs.sum(axis=1)
        class_probs = class_probs / row_sums[:, None]
        
        # Place evaluation points in their corresponding leaf node.
        # Store evaluation posterior in a num_eval-by-num_class matrix.
        eval_class_probs = class_probs[tree.apply(X[eval_indices])]
        # eval_class_probs = [class_probs[x] for x in tree.apply(X[eval_indices])]
        eval_entropies = [entropy(posterior) for posterior in eval_class_probs]
        cond_entropy += np.mean(eval_entropies)

      
    return cond_entropy / n_estimators
        
```


```python
def generate_data2(n, d, mu = 1):
    n_1 = np.random.binomial(n, .5) # number of class 1
    mean = np.zeros(d)
    mean[0] = mu
    X_1 = np.random.multivariate_normal(mean, np.eye(d), n_1)
    
    X = np.concatenate((X_1, np.random.multivariate_normal(-mean, np.eye(d), n - n_1)))
    y = np.concatenate((np.repeat(1, n_1), np.repeat(0, n - n_1)))
  
    return X, y
```


```python
def generate_data(n, d, mu = 1, var1 = 1, pi = 0.5, three_class = False):
    
    means, Sigmas, probs = _make_params(d, mu = mu, var1 = var1, pi = pi, three_class = three_class)
    counts = np.random.multinomial(n, probs, size = 1)[0]
    
    X_data = []
    y_data = []
    for k in range(len(probs)):
        X_data.append(np.random.multivariate_normal(means[k], Sigmas[k], counts[k]))
        y_data.append(np.repeat(k, counts[k]))
    X = np.concatenate(tuple(X_data))
    y = np.concatenate(tuple(y_data))
    
    return X, y
```


```python
def _make_params(d, mu = 1, var1 = 1, pi = 0.5, three_class = False):
    
    if three_class:
        return _make_three_class_params(d, mu, pi)
    
    mean = np.zeros(d)
    mean[0] = mu
    means = [mean, -mean]

    Sigma1 = np.eye(d)
    Sigma1[0, 0] = var1
    Sigmas = [np.eye(d), Sigma1]
    
    probs = [pi, 1 - pi]
    
    return means, Sigmas, probs

def _make_three_class_params(d, mu, pi):
    
    means = []
    mean = np.zeros(d)
    
    mean[0] = mu
    means.append(copy.deepcopy(mean))
    
    mean[0] = -mu
    means.append(copy.deepcopy(mean))
    
    mean[0] = 0
    mean[d-1] = mu
    means.append(copy.deepcopy(mean))
    
    Sigmas = [np.eye(d)]*3
    probs = [pi, (1 - pi) / 2, (1 - pi) / 2]
    
    return means, Sigmas, probs
```


```python
def compute_mutual_info(d, base = np.exp(1), mu = 1, var1 = 1, pi = 0.5, three_class = False):
    
    if d > 1:
        dim = 2
    else:
        dim = 1
 
    means, Sigmas, probs = _make_params(dim, mu = mu, var1 = var1, pi = pi, three_class = three_class)
    
    # Compute entropy and X and Y.
    def func(*args):
        x = np.array(args)
        p = 0
        for k in range(len(means)):
            p += probs[k] * multivariate_normal.pdf(x, means[k], Sigmas[k])
        return -p * np.log(p) / np.log(base)

    scale = 10
    lims = [[-scale, scale]]*dim
    H_X, int_err = nquad(func, lims)
    H_Y = entropy(probs, base = base)
    
    # Compute MI.
    H_XY = 0
    for k in range(len(means)):
        H_XY += probs[k] * (dim * np.log(2*np.pi) + np.log(np.linalg.det(Sigmas[k])) + dim) / (2 * np.log(base))
    I_XY = H_X - H_XY
    
    return I_XY, H_X, H_Y
```


```python
n = 20 
#n = 6000
mus = range(5)
ds = range(1, 16)
mu = 1
num_trials = 10
reps = 1
d = 2
pis = [0.05 * i for i in range(1, 20)]

def estimate_mi(X, y, est_H_Y, norm_factor): 
    return (est_H_Y - uf(np.array(X), y)) / norm_factor

def mi(X, y, n, d, pis, num_trials):
    #def worker(t): 
        #X, y = generate_data(n, d, pi = elem)
        
        #I_XY, H_X, H_Y = compute_mutual_info(d, pi = elem)
        I_XY, H_X, H_Y = compute_mutual_info(d)
        norm_factor = min(H_X, H_Y)
        
        _, counts = np.unique(y, return_counts=True)
        est_H_Y = entropy(counts, base=np.exp(1))
        ret = []
        ret.append(estimate_mi(X, y, est_H_Y, norm_factor))
        #return tuple(ret)
        return ret[0]
    
    #output = np.zeros((len(pis), num_trials))
    #for i, elem in enumerate(pis): 
        #results = np.array(Parallel(n_jobs=-2)(delayed(worker)(t) for t in range(num_trials)))
        #output[i, :] = results[:, 0]
    #return output
    
```


```python
def _perm_stat(X, y, is_distsim = True, permuter = None): 
    if permuter is None: 
        order = np.random.permutation(y.shape[0])
    else: 
        order = permuter()
    
    if is_distsim: 
        permy = y[order][:, order]
    else: 
        permy = y[order]
    
    perm_stat = mi(X, permy, n, d, pis, num_trials)
    
    return perm_stat
```


```python
def perm_test(X, y, workers = 1, is_distsim=True, perm_block = None): 

    # calculate observed test statistic
    stat = mi(X, y, n, d, pis, num_trials)
    print(stat) 

    # calculate null distribution
    null_dist = np.array(Parallel(n_jobs=-2)(delayed(_perm_stat)(X, y, is_distsim) for rep in range(reps)))
            
    pvalue = (null_dist >= stat).sum() / reps

    # correct for a p-value of 0. This is because, with bootstrapping
    # permutations, a p-value of 0 is incorrect
    if pvalue == 0:
        pvalue = 1 / reps

    return stat, pvalue, null_dist
 
```


```python
X, y = generate_data(n, d)
print(perm_test(X, y))
```

    0.2662616765000058
    


    ---------------------------------------------------------------------------

    _RemoteTraceback                          Traceback (most recent call last)

    _RemoteTraceback: 
    """
    Traceback (most recent call last):
      File "C:\Users\siptest\anaconda3\lib\site-packages\joblib\externals\loky\process_executor.py", line 418, in _process_worker
        r = call_item()
      File "C:\Users\siptest\anaconda3\lib\site-packages\joblib\externals\loky\process_executor.py", line 272, in __call__
        return self.fn(*self.args, **self.kwargs)
      File "C:\Users\siptest\anaconda3\lib\site-packages\joblib\_parallel_backends.py", line 608, in __call__
        return self.func(*args, **kwargs)
      File "C:\Users\siptest\anaconda3\lib\site-packages\joblib\parallel.py", line 256, in __call__
        for func, args, kwargs in self.items]
      File "C:\Users\siptest\anaconda3\lib\site-packages\joblib\parallel.py", line 256, in <listcomp>
        for func, args, kwargs in self.items]
      File "<ipython-input-17-54560ca488f1>", line 8, in _perm_stat
    IndexError: too many indices for array
    """

    
    The above exception was the direct cause of the following exception:
    

    IndexError                                Traceback (most recent call last)

    <ipython-input-21-eb21dea56cd2> in <module>
          1 X, y = generate_data(n, d)
    ----> 2 print(perm_test(X, y))
    

    <ipython-input-20-c30ec52f56db> in perm_test(X, y, workers, is_distsim, perm_block)
          6 
          7     # calculate null distribution
    ----> 8     null_dist = np.array(Parallel(n_jobs=-2)(delayed(_perm_stat)(X, y, is_distsim) for rep in range(reps)))
          9 
         10     pvalue = (null_dist >= stat).sum() / reps
    

    ~\anaconda3\lib\site-packages\joblib\parallel.py in __call__(self, iterable)
       1015 
       1016             with self._backend.retrieval_context():
    -> 1017                 self.retrieve()
       1018             # Make sure that we get a last message telling us we are done
       1019             elapsed_time = time.time() - self._start_time
    

    ~\anaconda3\lib\site-packages\joblib\parallel.py in retrieve(self)
        907             try:
        908                 if getattr(self._backend, 'supports_timeout', False):
    --> 909                     self._output.extend(job.get(timeout=self.timeout))
        910                 else:
        911                     self._output.extend(job.get())
    

    ~\anaconda3\lib\site-packages\joblib\_parallel_backends.py in wrap_future_result(future, timeout)
        560         AsyncResults.get from multiprocessing."""
        561         try:
    --> 562             return future.result(timeout=timeout)
        563         except LokyTimeoutError:
        564             raise TimeoutError()
    

    ~\anaconda3\lib\concurrent\futures\_base.py in result(self, timeout)
        433                 raise CancelledError()
        434             elif self._state == FINISHED:
    --> 435                 return self.__get_result()
        436             else:
        437                 raise TimeoutError()
    

    ~\anaconda3\lib\concurrent\futures\_base.py in __get_result(self)
        382     def __get_result(self):
        383         if self._exception:
    --> 384             raise self._exception
        385         else:
        386             return self._result
    

    IndexError: too many indices for array

