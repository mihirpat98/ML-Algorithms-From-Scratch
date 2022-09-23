import numpy as np
import matplotlib.pyplot as plt




"""
x_1 = Brown Color (A value of 1 indicates "Brown" cap color and 0 indicates "Red" cap color)
x_2 = Tapering Shape (A value of 1 indicates "Tapering Stalk Shape" and 0 indicates "Enlarging" stalk shape)
x_3 = Solitary (A value of 1 indicates "Yes" and 0 indicates "No")
y = 1 indicates edible
y = 0 indicates poisonous
"""
X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])



def compute_entropy(y):
   
    entropy = 0.
    p1 = 0
    
    if len(y) != 0:
        p1 = len(y[y == 1]) / len(y) 
        if p1!=0 and p1!=1:
            entropy = -p1 * np.log2(p1) - (1-p1) * np.log2(1-p1)
    else:
        pass    
    
    return entropy

def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (ndarray):  List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (ndarray): Indices with feature value == 1
        right_indices (ndarray): Indices with feature value == 0
    """
    left_indices = []
    right_indices = []
    for i in node_indices:
        if X[i][feature] ==1:
            left_indices.append(i)
        else:
            right_indices.append(i)    
    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
   
    Returns:
        cost (float):        Cost computed
    
    """    
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    information_gain = 0

    H_p1_node = compute_entropy(y_node)
    H_p1_left = compute_entropy(y_left)
    H_p1_right = compute_entropy(y_right)
    w_left = len(X_left)/(len(X_node))
    w_right = len(X_right)/(len(X_node))
    information_gain =   H_p1_node - ((w_left*H_p1_left)+ (w_right*H_p1_right))
    return information_gain






root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)

info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)



# Get Best Split based on lowest info gain
def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    """    
    
    num_features = X.shape[1]
    
    best_feature = -1
    info_gain_max = 0
    for i in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, i)
        if info_gain>info_gain_max:
            best_feature = i
            info_gain_max = info_gain   
    return best_feature

best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)


#Building the tree

tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)

build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
