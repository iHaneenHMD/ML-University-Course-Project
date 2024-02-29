# ML-University-Course-Project
implement from scratch the K-mean algorithm and DBSCAN . Then Create datasets . Show a comparison between the results of the two models.

Generate 4 dataset :
Dataset1: blobs dataset
X,y = datasets.make_blobs(n_samples=n_samples,random_state=random_state)
Dataset2: Anisotropicly distributed dataset
X, _ = datasets.make_blobs(n_samples=n_samples,random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X = np.dot(X, transformation)
Dataset3: noisy moons dataset
X, y = datasets.make_moons(n_samples=n_samples, noise=0.1,random_state=random_state)
Dataset4: noisy circles dataset
X,y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05,random_state=random_state)
