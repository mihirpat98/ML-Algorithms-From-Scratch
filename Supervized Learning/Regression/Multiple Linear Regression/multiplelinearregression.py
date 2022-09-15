import numpy as np
import copy

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])


#  Z Score normalization
def zscore_normalize_features(X):
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      
    return X_norm

#X_train = zscore_normalize_features(X_train)

def predict(x,w,b):
    return np.dot(x,w)+b

def compute_cost(X,y,w,b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i],w) + b
        cost += (f_wb_i-y[i])**2
    cost = cost/(2*m)
    return cost


def compute_gradient(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot(X[i],w)+b) -y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err*X[i,j]
        dj_db = dj_db + err
    dj_db = dj_db/m
    dj_dw = dj_dw/m

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db,dj_dw = gradient_function(X,y,w,b)
        w = w-alpha*dj_dw
        b = b -alpha*dj_db

    return w,b

initial_w = np.zeros_like(w_init) 
initial_b = 0.
iterations = 10000
alpha = 5.0e-7
w_final, b_final = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)

print(w_final,b_final)

# Make Prediction
y_pred = np.dot(X_train,w_final) + b_final
print(y_pred,y_train)

