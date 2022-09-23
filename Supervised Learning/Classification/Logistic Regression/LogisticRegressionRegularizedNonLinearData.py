import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy

X_train = np.array([[0.051267,0.69956], [-0.092742,0.68494], [-0.21371,0.69225], [-0.375,0.50219], [-0.51325,0.46564], [-0.52477,0.2098], [-0.39804,0.034357], [-0.30588,-0.19225], [0.016705,-0.40424], [0.13191,-0.51389], [0.38537,-0.56506], [0.52938,-0.5212], [0.63882,-0.24342], [0.73675,-0.18494], [0.54666,0.48757], [0.322,0.5826], [0.16647,0.53874], [-0.046659,0.81652], [-0.17339,0.69956], [-0.47869,0.63377], [-0.60541,0.59722], [-0.62846,0.33406], [-0.59389,0.005117], [-0.42108,-0.27266], [-0.11578,-0.39693], [0.20104,-0.60161], [0.46601,-0.53582], [0.67339,-0.53582], [-0.13882,0.54605], [-0.29435,0.77997], [-0.26555,0.96272], [-0.16187,0.8019], [-0.17339,0.64839], [-0.28283,0.47295], [-0.36348,0.31213], [-0.30012,0.027047], [-0.23675,-0.21418], [-0.06394,-0.18494], [0.062788,-0.16301], [0.22984,-0.41155], [0.2932,-0.2288], [0.48329,-0.18494], [0.64459,-0.14108], [0.46025,0.012427], [0.6273,0.15863], [0.57546,0.26827], [0.72523,0.44371], [0.22408,0.52412], [0.44297,0.67032], [0.322,0.69225], [0.13767,0.57529], [-0.0063364,0.39985], [-0.092742,0.55336], [-0.20795,0.35599], [-0.20795,0.17325], [-0.43836,0.21711], [-0.21947,-0.016813], [-0.13882,-0.27266], [0.18376,0.93348], [0.22408,0.77997], [0.29896,0.61915], [0.50634,0.75804], [0.61578,0.7288], [0.60426,0.59722], [0.76555,0.50219], [0.92684,0.3633], [0.82316,0.27558], [0.96141,0.085526], [0.93836,0.012427], [0.86348,-0.082602], [0.89804,-0.20687], [0.85196,-0.36769], [0.82892,-0.5212], [0.79435,-0.55775], [0.59274,-0.7405], [0.51786,-0.5943], [0.46601,-0.41886], [0.35081,-0.57968], [0.28744,-0.76974], [0.085829,-0.75512], [0.14919,-0.57968], [-0.13306,-0.4481], [-0.40956,-0.41155], [-0.39228,-0.25804], [-0.74366,-0.25804], [-0.69758,0.041667], [-0.75518,0.2902], [-0.69758,0.68494], [-0.4038,0.70687], [-0.38076,0.91886], [-0.50749,0.90424], [-0.54781,0.70687], [0.10311,0.77997], [0.057028,0.91886], [-0.10426,0.99196], [-0.081221,1.1089], [0.28744,1.087], [0.39689,0.82383], [0.63882,0.88962], [0.82316,0.66301], [0.67339,0.64108], [1.0709,0.10015], [-0.046659,-0.57968], [-0.23675,-0.63816], [-0.15035,-0.36769], [-0.49021,-0.3019], [-0.46717,-0.13377], [-0.28859,-0.060673], [-0.61118,-0.067982], [-0.66302,-0.21418], [-0.59965,-0.41886], [-0.72638,-0.082602], [-0.83007,0.31213], [-0.72062,0.53874], [-0.59389,0.49488], [-0.48445,0.99927], [-0.0063364,0.99927], [0.63265,-0.030612]])
y_train = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


colors = ['red','green']
x1= []
x2 = []
for i in range(len(X_train)):
  x1.append(X_train[i][0])
  x2.append(X_train[i][1])
plt.scatter(x1,x2,c= y_train,cmap=matplotlib.colors.ListedColormap(colors),marker ='x')
plt.title('Logistic Regression')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()







def map_feature1(x1, x2):
    '''
    Maps the two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    Inputs X1, X2 must be the same size
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(shape=(x1[:, 0].size, 1))

    m, n = out.shape

    for i in range(1, degree + 1):
        
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)
    return out

def map_feature(X1, X2, degree):
    res = np.ones(X1.shape[0])
    for i in range(1,degree + 1):
        for j in range(0,i + 1):
            res = np.column_stack((res, (X1 ** (i-j)) * (X2 ** j)))
    return res
def sigmoid(a):
    return 1/(1+np.exp(-a))


def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost

def compute_cost_reg(X, y, w, b, lambda_ = 1):
    

    m, n = X.shape
    
    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost_logistic(X, y, w, b) 
    
    # You need to calculate this value
    reg_cost = 0.
    
    ### START CODE HERE ###
    for i in range(n):
        reg_cost += (w[i])**2
    
    ### END CODE HERE ### 
    
    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + (lambda_/(2 * m)) * reg_cost

    return total_cost

def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    m,n = X.shape
    dj_dw = np.zeros((n,))                            #(n,)
    dj_db = 0.0                                       #scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw  



def gradient_descent(X, y, w_in, b_in, alpha, num_iters,lambda_): 
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic_reg(X, y, w, b,lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db  
    return w,b

X_mapped = map_feature(X_train[:, 0], X_train[:, 1],6)
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
iterations = 10000
alpha = 0.01
final_w,final_b = gradient_descent(X_mapped, y_train, initial_w, initial_b, alpha,iterations,lambda_)
print(final_w,final_b)


def predict(X, w, b): 

    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
   
    ### START CODE HERE ### 
    # Loop over each example
    for i in range(m):   
        z_wb = 0
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb += X[i,j]*w[j]
        
        # Add bias term 
        z_wb += b
        
        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)

        # Apply the threshold
        p[i] = 0 if f_wb<0.5 else 1
        
    ### END CODE HERE ### 
    return p



p = predict(X_mapped, final_w, final_b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))