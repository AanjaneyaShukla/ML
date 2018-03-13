# imports go here
import numpy as np


def filtering_path(transition_matrix, evidence_matrix, init_prob, forward_count, observation):
    for i in range(forward_count):
        forward_matrix = np.multiply(np.array(evidence_matrix[:, observation[i]]).reshape(2, 1),
                                     np.matmul(transition_matrix, init_prob))
        #forward_matrix[:] = forward_matrix[:] / np.sum(forward_matrix[:])
        print forward_matrix
        init_prob = forward_matrix
    return forward_matrix


# def backward_path(transition_matrix, evidence_matrix, init_prob, backward_count, observation):
#    for

def predict_path(transition_matrix, init_prob, predict_count):
    for i in range(predict_count):
        predict_matrix = np.matmul(transition_matrix, init_prob)
        print predict_matrix
        init_prob = predict_matrix
    return predict_matrix


def q3(transition_matrix, new_transition_matrix, evidence_matrix, init_prob, observations):
    forward_count = 6
    predict_count = 6
    filtering_res = filtering_path(transition_matrix, evidence_matrix, init_prob, forward_count, observations)
    predict_path(new_transition_matrix, filtering_res, predict_count)


def smoothing(transition_matrix, evidence_matrix, init_prob, observations):
    res = np.ones((2, 1))
    for i in range(len(observations)):
        res = np.multiply(res, np.matmul(transition_matrix,
                                         np.array(evidence_matrix[:, observations[len(observations) - i - 1]]).reshape(
                                             2, 1)))
        print res


def init():
    transition_matrix = np.array([[.75, .25], [1 - .75, 1 - .25]])
    evidence_matrix = np.array([[.71, 1 - .71], [.01, 1 - .01]])
    init_prob = np.array([[.2], [1 - .2]])
    # 0-True; 1-False
    observations = [0, 1, 0, 1, 0, 1]
    new_transition_matrix = np.array([[.81, .16], [1 - .81, 1 - .16]])
    q3(transition_matrix, new_transition_matrix, evidence_matrix, init_prob, observations)
    #smoothing(transition_matrix, evidence_matrix, init_prob, observations)

# main script stuff goes here
if __name__ == '__main__':
    init()