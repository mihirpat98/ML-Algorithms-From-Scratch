import numpy as np
import matplotlib.pyplot as plt
import matplotlib

original_img = plt.imread('bird_small.png')
plt.imshow(original_img)
plt.show()
original_img = original_img / 255
# Reshape the image into an m x 3 matrix where m = number of pixels
# (in this case m = 128 x 128 = 16384)
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X_img that we will use K-Means on.

X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))
print(X_img.shape)
def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    """
    # Set K
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for i,x in enumerate(X):
        distance = []
        for n in centroids:
            norm_ij = np.linalg.norm(x-n)
            distance.append(norm_ij)
        idx[i]=np.argmin(distance)
    return idx


def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the 
    data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each 
                       example in X. Concretely, idx[i] contains the index of 
                       the centroid closest to example i
        K (int):       number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    # Useful variables
    m, n = X.shape
    # You need to return the following variables correctly
    centroids = np.zeros((K, n))
    ### START CODE HERE ###
    for i in range(K):
        
        centroids[i] = np.sum(X[idx==i],axis = 0)/(len(X[idx==i])+0.0000001)
    return centroids


def run_kMeans(X, initial_centroids, max_iters=10):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """
    
    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    idx = np.zeros(m)
    # Run K-Means
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    
    return centroids, idx




# To initialize centroids randomly
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be 
    used in K-Means on the dataset X
    
    Args:
        X (ndarray): Data points 
        K (int):     number of centroids/clusters
    
    Returns:
        centroids (ndarray): Initialized centroids
    """
    
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    
    # Take the first K examples as centroids
    centroids = X[randidx[:K]]
    
    return centroids






# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16                       
max_iters = 10               

# Using the function you have implemented above. 
initial_centroids = kMeans_init_centroids(X_img, K) 

# Run K-Means - this takes a couple of minutes
centroids, idx = run_kMeans(X_img, initial_centroids, max_iters) 

print("Shape of idx:", idx.shape)
print("Closest centroid for the first five elements:", idx[:5])

# Represent image in terms of indices
X_recovered = centroids[idx, :] 

# Reshape recovered image into proper dimensions
X_recovered = np.reshape(X_recovered, original_img.shape) 

# Display original image
fig, ax = plt.subplots(1,2, figsize=(8,8))
plt.axis('off')

ax[0].imshow(original_img*255)
ax[0].set_title('Original')
ax[0].set_axis_off()



# Display compressed image
ax[1].imshow(X_recovered*255)
ax[1].set_title('Compressed with %d colours'%K)
ax[1].set_axis_off()
plt.show()


# REDUCES IMAGE SIZE BY FACTOR OF 6
'''
After finding the top K=16 colors to represent the image, you can now assign each pixel position to its closest centroid using the `find_closest_centroids` function. 
* This allows you to represent the original image using the centroid assignments of each pixel. 
* Notice that you have significantly reduced the number of bits that are required to describe the image. 
    * The original image required 24 bits for each one of the 128X128 pixel locations, resulting in total size of 128X128X24 = 393,216 bits. 
    * The new representation requires some overhead storage in form of a dictionary of 16 colors, each of which require 24 bits, but the image itself then only requires 4 bits per pixel location. 
    * The final number of bits used is therefore 16X24 + 128X128X4 = 65,920 bits, which corresponds to compressing the original image by about a factor of 6.'''