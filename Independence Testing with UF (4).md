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
    def worker(t): 
        #X, y = generate_data(n, d, pi = elem)
        
        I_XY, H_X, H_Y = compute_mutual_info(d, pi = elem)
        norm_factor = min(H_X, H_Y)
        
        _, counts = np.unique(y, return_counts=True)
        est_H_Y = entropy(counts, base=np.exp(1))
        ret = []
        ret.append(estimate_mi(X, y, est_H_Y, norm_factor))
        return tuple(ret)
    output = np.zeros((len(pis), num_trials))
    for i, elem in enumerate(pis): 
        results = np.array(Parallel(n_jobs=-2)(delayed(worker)(t) for t in range(num_trials)))
        output[i, :] = results[:, 0]
    return output
    
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
    null_dist = np.array(
        Parallel(n_jobs=-2)
        [
            delayed(_perm_stat)(X, y, is_distsim) 
            for rep in range(reps)
        ]
    )
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

    [[0.37124989 0.39801119 0.32884694 0.34096027 0.35788863 0.33299354
      0.330759   0.37953134 0.39087074 0.31025029]
     [0.24151975 0.24114328 0.20138225 0.24383607 0.20328413 0.24996326
      0.22394985 0.2259126  0.24311177 0.22016077]
     [0.16585358 0.1858278  0.17617689 0.18075625 0.17697    0.15283317
      0.19609625 0.16835676 0.15807554 0.19941406]
     [0.13192116 0.15627886 0.14502141 0.15052429 0.16664959 0.13707237
      0.14167613 0.11907549 0.14037807 0.14599692]
     [0.1390065  0.12514073 0.13576076 0.12204698 0.1402708  0.13541032
      0.12684372 0.1161537  0.11764514 0.12911777]
     [0.10592341 0.12037681 0.10180546 0.12019028 0.10982195 0.1181776
      0.10775551 0.10875831 0.11305231 0.12519944]
     [0.09162848 0.101524   0.12040745 0.10638535 0.11761253 0.11885016
      0.10780773 0.10814951 0.11369171 0.10844179]
     [0.10640369 0.11781154 0.10277303 0.10543168 0.10022633 0.11287568
      0.11774251 0.11243468 0.10500478 0.09947607]
     [0.1110501  0.1041718  0.09614161 0.09088331 0.10729562 0.10962854
      0.09844251 0.09183759 0.10048592 0.12170865]
     [0.09497075 0.10615765 0.10155557 0.11597628 0.1011116  0.11048117
      0.10736397 0.09685185 0.10983524 0.1194312 ]
     [0.10621773 0.10546786 0.08888219 0.09994732 0.10399894 0.10692484
      0.10371809 0.10739096 0.11771101 0.12326399]
     [0.11455663 0.11666259 0.08913673 0.11413356 0.10929592 0.10215871
      0.09812423 0.11649767 0.12241687 0.10517491]
     [0.1095785  0.10967638 0.11427874 0.10777136 0.10882647 0.12492731
      0.12035367 0.1060403  0.09980573 0.10240209]
     [0.11943186 0.11676421 0.10904836 0.11742864 0.1116706  0.12127662
      0.11456932 0.08725354 0.12003814 0.11716401]
     [0.12590515 0.1216024  0.12602665 0.14000881 0.12614742 0.14017163
      0.12186262 0.12972327 0.12296164 0.13159339]
     [0.1354202  0.13913553 0.16090152 0.14534251 0.13558759 0.15656972
      0.14714033 0.13335306 0.15116954 0.1315072 ]
     [0.1653937  0.15877144 0.18038406 0.16709768 0.17308607 0.18826174
      0.15880121 0.15745216 0.16509815 0.17970539]
     [0.2603865  0.24459816 0.19226772 0.21648894 0.24061613 0.21792439
      0.20940791 0.20085369 0.22416092 0.20603499]
     [0.37514313 0.31018483 0.3722289  0.3080559  0.36124659 0.32643893
      0.39587829 0.3685064  0.37788491 0.35847521]]
    


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
      File "<ipython-input-8-54560ca488f1>", line 8, in _perm_stat
    IndexError: too many indices for array
    """

    
    The above exception was the direct cause of the following exception:
    

    IndexError                                Traceback (most recent call last)

    <ipython-input-10-eb21dea56cd2> in <module>
          1 X, y = generate_data(n, d)
    ----> 2 print(perm_test(X, y))
    

    <ipython-input-9-e3aba0e84ccb> in perm_test(X, y, workers, is_distsim, perm_block)
          7     # calculate null distribution
          8     null_dist = np.array(
    ----> 9         Parallel(n_jobs=-2)(delayed(_perm_stat)(X, y, is_distsim) for rep in range(reps))
         10     )
         11     pvalue = (null_dist >= stat).sum() / reps
    

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

