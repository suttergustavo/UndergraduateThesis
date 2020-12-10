import numpy as np
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from sklearn.cluster import AgglomerativeClustering

eps = 1e-16

class InceptionScore:
    def __init__(self, verbose=True, lat_as_input=False):
        if not lat_as_input:
            self.model = InceptionV3()
        self.verbose = 1 if verbose else 0
        self.lat_as_input = lat_as_input

    def evaluate(self, X):
        if self.lat_as_input:
            p_yx = X
        else:
            p_yx = self.model.predict_generator(X, verbose=self.verbose)
        
        p_y = np.mean(p_yx, axis=0)

        kl = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        kl = np.sum(kl, axis=1)

        expected_value = np.mean(kl)

        score = np.exp(expected_value)

        return score


class FID:
    def __init__(self, verbose=True, lat_as_input=False):
        if not lat_as_input:
            self.model = InceptionV3(include_top=False, pooling='avg')
    
        self.verbose = 1 if verbose else 0
        self.lat_as_input = lat_as_input

    def fit(self, X_real):
        if self.lat_as_input:
            self.lat_real = X_real
        else:
            self.lat_real = self.model.predict_generator(X_real, verbose=self.verbose)
        self.mu_real = np.mean(self.lat_real, axis=0)
        self.sigma_real = np.cov(self.lat_real, rowvar=False)

    def evaluate(self, X_fake):
        if self.lat_as_input:
            lat_fake = X_fake
        else:
            lat_fake = self.model.predict_generator(X_fake, verbose=self.verbose)
        
        mu_fake = np.mean(lat_fake, axis=0)
        sigma_fake = np.cov(lat_fake, rowvar=False)

        mu_diff = np.sum(np.square(self.mu_real - mu_fake))
        cov_sqrt = sqrtm(np.dot(self.sigma_real, sigma_fake))

        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
            
        fid = mu_diff + np.trace(self.sigma_real + sigma_fake - 2 * cov_sqrt)

        return fid

class DendrogramDistance:
    def __init__(self, use_layer='last', agg_type='max', verbose=True):
        if use_layer == 'last':
            self.model = InceptionV3()
        elif use_layer == 'hidden':
            self.model = InceptionV3(include_top=False, pooling='avg')

        self.agg_type = agg_type
        self.verbose = verbose
        self.use_layer = use_layer

    def fit(self, X_true, return_lat=False):
        if self.use_layer != 'pixel':
            self.lat_real = self.model.predict_generator(X_true, verbose=self.verbose)
        else:
            self.lat_real = next(X_true)[0].reshape(X_true.batch_size, -1)
            for _ in range(len(X_true)-1):
                batch = next(X_true)[0]
                batch = batch.reshape(batch.shape[0], -1)
                self.lat_real = np.concatenate((self.lat_real, batch))
        
        self.dendro_real = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='single')
        self.dendro_real.fit(self.lat_real)
        
        if return_lat:
            return self.lat_real

    def evaluate(self, X_fake, return_lat=False):
        if self.use_layer != 'pixel':
            lat_fake = self.model.predict_generator(X_fake, verbose=self.verbose)
        else:
            lat_fake = next(X_fake)[0].reshape(X_fake.batch_size, -1)
            for _ in range(len(X_fake)-1):
                batch = next(X_fake)[0]
                batch = batch.reshape(batch.shape[0], -1)
                lat_fake = np.concatenate((lat_fake, batch))

        self.dendro_fake = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage='single')
        self.dendro_fake.fit(lat_fake)
        
        d_real = self.dendro_real.distances_
        d_fake = self.dendro_fake.distances_

        score = self.gh_distance(d_real, d_fake, agg=self.agg_type)

        if return_lat:
            return score, lat_fake
        return score
        
    
    def gh_distance(self, d1, d2, agg='max'):
        max_d = max(d1.shape[0], d2.shape[0])
    
        alpha = np.zeros(max_d)
        alpha[:d1.shape[0]] = d1
    
        beta = np.zeros(max_d)
        beta[:d2.shape[0]] = d2
    
        alpha = np.sort(alpha)
        beta = np.sort(beta)

        if agg == 'max':
            return np.max(np.abs(alpha - beta))
        elif agg == 'mean':
            return np.mean(np.abs(alpha - beta))
        elif agg == 'both':
            return {'max': np.max(np.abs(alpha - beta)),
                    'mean': np.mean(np.abs(alpha - beta))}