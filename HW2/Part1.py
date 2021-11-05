import numpy as np
from collections import defaultdict

# region DATA
# create train sample
X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]
])
# create label for our train sample
Y_train = np.array(['Y', 'N', 'Y', 'Y'])

# create test case model
X_test = np.array([[1, 1, 0]])


# endregion

def get_label_indices(labels):
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices


def get_prior(label_indices):
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior


def get_likelihood(features, label_indices, smoothing=0):
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        print(total_count)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    return likelihood


def get_posterior(X, prior, likelihood):
    posteriors = []
    for x in X:
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])
            sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors


print(get_posterior(X_test, get_prior(get_label_indices(Y_train)),
                    get_likelihood(X_train, get_label_indices(Y_train), 1)))
