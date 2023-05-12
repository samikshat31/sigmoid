import numpy as np

def sigmoid(x) :
    return 1.0 / (1.0 + np.exp(-x))

m = np.array([ [ 9.6, 3.3, 5.6, 7.2, 5.6, 5.4, 1.9, 8.2, 10. , 7.1],
               [ 1. , 4.7, 0.8, 9.1, 7.6, 2.1, 4.8, 2.5, 7.4, 3.8],
               [ 6.6, 7. , 0.6, 2. , 4. , 9.1, 7.9, 6.9, 9.7, 3.6],
               [ 9.1, 5.9, 8.4, 7.9, 7.8, 6.8, 6.3, 0.9, 1.4, 5.3] ])

epsilon = 1e-5
gamma = 4
beta = 0.3

# calculate the mean and variance of each row
row_means = np.mean(m, axis=1)
row_variances = np.var(m, axis=1)
print("BATCH NORMALISATION")


# perform mean normalization and variance scaling for each row
for i in range(m.shape[0]):
    m_norm = gamma*((m[i] - row_means[i]) / np.sqrt(row_variances[i] + epsilon)) + beta
    n = sigmoid(m_norm)
    print("Normalized row", i, ":", n)
    print(" ")
    
print(" ")
# calculate the mean and variance of each column
col_means = np.mean(m, axis=0)
col_variances = np.var(m, axis=0)

print("LAYER NORMALISATION")
# perform mean normalization and variance scaling for each column
for i in range(m.shape[0]):
    m_norm = gamma*((m[i] - col_means[i]) / np.sqrt(col_variances[i] + epsilon)) + beta
    n = sigmoid(m_norm)
    print("Normalized row", i, ":", n)
    print(" ")
