# encoding=utf-8
import numpy as np
from sklearn.utils import check_X_y, check_array
from sklearn.utils.fixes import logsumexp

THRESHOLD = 1e-04


class NaiveBayes(object):
    def __init__(self, is_feature_continuous=None, priors=None, debug=False):
        """Init.

        :param is_feature_continuous: discrete/continuous
        :param priors: priors of classes
        """
        self.is_feature_continuous = is_feature_continuous
        self.priors = priors
        self.is_fitted = False
        self.debug = debug

    def _update_mean_variance(self, n_past, mu, var, conditional_p, X):
        """计算每个类下的条件概率分布"""
        if X.shape[0] == 0:
            return mu, var, conditional_p

        if self.debug:
            if n_past != 0:
                print 'ERROR: n_past = {}, while it is expected to be 0'.format(n_past)
        else:
            assert n_past == 0
        new_var = np.var(X, axis=0)
        new_mu = np.mean(X, axis=0)
        n_features = X.shape[1]
        new_conditional_p = []
        for i in range(n_features):
            if self.is_feature_continuous[i]:
                new_conditional_p.append({})
                continue
            feature_values = X[:, i]
            values, counts = np.unique(feature_values, return_counts=True)
            # sum_ = np.sum(counts)
            d = dict()
            for j, v in enumerate(values):
                v = int(v)
                if self.debug:
                    if v in d:
                        print 'ERROR: v in d, while it is expected not in d'
                else:
                    assert v not in d
                d[v] = counts[j]
            # if self.debug:
            #     if abs(np.sum(d.values()) - 1) > THRESHOLD:
            #         print 'WARN: np.sum(d.values()) = {}, delta with 1 is {}, while it is expected to be 1'.format(
            #             np.sum(d.values()), np.sum(d.values()) - 1)
            # else:
            #     assert np.sum(d.values()) == 1
            new_conditional_p.append(d)
        new_conditional_p = np.array(new_conditional_p)
        return new_mu, new_var, new_conditional_p

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        epsilon = 1e-9 * np.var(X, axis=0).max()

        classes = np.unique(y)
        self.classes_ = classes
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        if self.is_feature_continuous is None:
            self.is_feature_continuous = [0] * n_features
        self.is_feature_continuous = np.array(self.is_feature_continuous)
        assert self.is_feature_continuous.shape[0] == n_features

        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.conditional_p_ = np.array([[{} for v in range(n_features)] for u in range(n_classes)])

        self.class_count_ = np.zeros(n_classes, dtype=np.float64)

        # Initialise the class prior
        if self.priors is not None:
            priors = np.asarray(self.priors)
            # Check that the provide prior match the number of classes
            if len(priors) != n_classes:
                raise ValueError('Number of priors must match number of classes.')
            # Check that the sum is 1
            if priors.sum() != 1.0:
                raise ValueError('The sum of the priors should be 1.')
            # Check that the prior are non-negative
            if (priors < 0).any():
                raise ValueError('Priors must be non-negative.')
            self.class_prior_ = priors
        else:
            # Initialize the priors to zeros for each class
            self.class_prior_ = np.zeros(len(self.classes_), dtype=np.float64)

        unique_y = np.unique(y)
        for y_i in unique_y:
            i = classes.searchsorted(y_i)
            X_i = X[y == y_i, :]
            N_i = X_i.shape[0]

            new_theta, new_sigma, new_conditional_p = self._update_mean_variance(
                self.class_count_[i], self.theta_[i, :], self.sigma_[i, :],
                self.conditional_p_[i, :], X_i)

            self.theta_[i, :] = new_theta
            self.sigma_[i, :] = new_sigma
            self.conditional_p_[i, :] = new_conditional_p
            self.class_count_[i] += N_i

        self.sigma_[:, :] += epsilon

        # Update if only no priors is provided
        if self.priors is None:
            # Empirical prior
            self.class_prior_ = self.class_count_ / self.class_count_.sum()

        self.is_fitted = True

        return self

    def _joint_log_likelihood(self, X):
        """计算log(likelihood * class_prior)"""
        assert self.is_fitted

        if self.debug:
            n_features = X.shape[1]
            n_classes = len(self.classes_)
            self.missing_feature_values = [[set() for v in range(n_features)] for u in range(n_classes)]

        X = check_array(X)

        joint_log_likelihood_for_all_X = []
        for x_idx, x in enumerate(X):
            # 对每一行分别计算
            joint_log_likelihood = []
            for i in range(np.size(self.classes_)):
                # 先验概率
                jointi = np.log(self.class_prior_[i])
                # 每个特征属性划分对第i个类别的条件概率估计的和
                for k, is_continuous in enumerate(self.is_feature_continuous):
                    if is_continuous:
                        # 按高斯分布算
                        # if self.debug:
                        #     print 'feature {}, continuous value: {}'.format(k, x[k])
                        n_ij = - 0.5 * np.log(2. * np.pi * self.sigma_[i, k])
                        n_ij -= 0.5 * ((x[k] - self.theta_[i, k]) ** 2) / (self.sigma_[i, k])
                        jointi += n_ij
                    else:
                        # 按离散值算
                        # if self.debug:
                        #     print 'feature {}, discrete value: {}'.format(k, x[k])
                        conditional_p = self.conditional_p_[i, k]
                        if self.debug:
                            if int(x[k]) not in conditional_p:
                                self.missing_feature_values[i][k].add(int(x[k]))
                                # print '{}th row data ==> {}th feature value {} not in train data'.format(
                                #     x_idx, k, int(x[k]))
                        if int(x[k]) not in conditional_p:
                            jointi += np.log(0.00000001)
                        else:
                            sum = np.sum(conditional_p.values())
                            jointi += np.log(conditional_p[int(x[k])] / float(sum))
                joint_log_likelihood.append(jointi)
            joint_log_likelihood_for_all_X.append(joint_log_likelihood)
        joint_log_likelihood_for_all_X = np.array(joint_log_likelihood_for_all_X)
        assert joint_log_likelihood_for_all_X.shape == (X.shape[0], len(self.classes_))
        return joint_log_likelihood_for_all_X

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))
