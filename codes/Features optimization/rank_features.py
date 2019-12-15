"""
Plots the features in their importance order to determine if it is possible
to take only a subset of the features into account

"""
import numpy as np
import matplotlib.pyplot as plt


def rank_features(estimator, X):

    importances = (estimator.feature_importances_)/max(estimator.feature_importances_)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the estimator
    plt.figure()

    plt.bar(range(X.shape[1]), importances[indices],
           color="r")
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticklabels([])
    plt.xlabel('Features')
    plt.ylabel('Features importances')
    plt.xlim([-1, X.shape[1]])
    plt.savefig("%s.pdf" % 'features_importances_123')
    plt.show()

    return indices