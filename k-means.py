import numpy as np

def intit_cent(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def assing_cluster(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis = 1)

def update_centroids(X, assignments, k):
    return np.array([X[assignments == i].mean(axis=0) for i in range(k)])

def k_mean(X, k):
    centroids = intit_cent(X, k)
    assigments = assing_cluster(X, centroids)
    prev_assigments = None

    #заменил while True на что-то более разумное
    for i in range(len(X)):

        if np.array_equal(assigments, prev_assigments):
            break

        centroids = update_centroids(X, assigments, k)
        prev_assigments = assigments

    return centroids, assigments


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x = x[:, np.newaxis]
k = 2
centroids, assignments = k_mean(x, k)

print("Центроиды:", centroids)
print("Кластеры:", assignments)