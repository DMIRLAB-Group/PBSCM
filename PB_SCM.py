from util import *
from tqdm import tqdm
from copy import deepcopy


class PB_SCM(object):
    def __init__(self, data, seed=1):
        self.n = len(data)
        self.sample_size = len(data[0])
        self.data = data
        self.skeleton_mat = None
        self.coeff_list = None
        self.record = [{} for i in range(self.n)]  # record the likelihood

        if seed is not None:
            self.rand_state = np.random.RandomState(seed)
            np.random.seed(seed)
        else:
            self.rand_state = np.random.RandomState()

    def dfs(self, graph, x, vis):
        if vis[x]:
            return True
        vis[x] = 1
        ret = False
        for i in range(self.n):
            if graph[x][i] == 1:
                ret = ret or self.dfs(graph, i, vis)
        vis[x] = 0
        return ret

    def check_circle(self, graph):
        n = self.n
        ret = False
        vis = np.zeros(n)
        for i in range(n):
            ret = ret or self.dfs(graph, i, vis)
        return ret

    def one_step_change_iterator(self, edge_mat):
        return map(lambda e: one_step_change(edge_mat, e),
                   product(range(self.n), range(self.n)))

    # return the list of parents of node x
    def get_parent(self, graph, x):
        parent_list = []
        for i in range(self.n):
            if graph[i, x]:
                parent_list.append(i)
            else:
                continue
        return np.array(parent_list)

    # estimate the coefficient
    def get_coefficient(self, parent_index, target_index):
        coeff = np.zeros((self.n+1))
        target_data = self.data[target_index]

        if len(parent_index) == 0:
            lam_for_noise = np.mean(target_data)
            coeff[-1] = lam_for_noise
            return coeff

        parent_data_list = self.data[parent_index]
        n = len(parent_data_list)
        if n == 1:
            parent_coeff = np.cov(parent_data_list[0], target_data)[0, 1] / np.mean(parent_data_list[0])
            parent_coeff = np.array([parent_coeff])
        else:
            cov_mat = np.cov(parent_data_list)
            cov_vec = np.zeros([n, 1])
            for i in range(n):
                cov_vec[i] = np.cov([parent_data_list[i], target_data])[0, 1]
            parent_coeff = np.dot(np.linalg.inv(cov_mat), cov_vec).ravel()

        E_of_preant = 0
        for i in range(n):
            E_of_preant += parent_coeff[i] * np.mean(parent_data_list[i])
        lam_for_noise = np.mean(target_data) - E_of_preant
        coeff[parent_index] = parent_coeff
        coeff[-1] = lam_for_noise
        return coeff

    def get_node_likelihood(self, coeff_list, parent_list, target_node):
        not_parent = np.delete(np.array([i for i in range(self.n)]), list(parent_list))
        samples = deepcopy(self.data)
        target_sample = deepcopy(samples[target_node, :])
        samples[not_parent, :] = 0
        samples = np.concatenate([samples, [target_sample]]).T

        sample_unique, unique_count = np.unique(samples, axis=0, return_counts=True)

        likelihood = 0.
        # calculate the likelihood of each sample
        for i in range(len(sample_unique)):
            Li_for_one_sample = get_sample_likelihood(coeff=coeff_list, parent_list=parent_list, sample=sample_unique[i])
            if Li_for_one_sample <= 0:
                likelihood += 0
            else:
                likelihood += np.log(Li_for_one_sample) * unique_count[i]
        return likelihood

    def get_total_likelihood(self, graph):
        likelihood = 0
        n, m = self.n, self.sample_size
        coeff = np.zeros((n, n + 1))  # coeff set
        if self.check_circle(graph):  # if not DAG
            return -np.inf, coeff

        for i in range(n):
            # get the parents of vertex i
            parent_list = self.get_parent(graph, i)
            parent_list_str = np.array2string(parent_list)
            if parent_list_str in self.record[i]:
                likelihood += self.record[i][parent_list_str][0]
                coeff[i] = self.record[i][parent_list_str][1]
                continue
            # Estimating coefficients
            coeff[i] = self.get_coefficient(parent_list, i)

            if len(parent_list) != 0 and \
                    ((coeff[i][parent_list] < 0).any() or
                     (coeff[i][parent_list] > 1).any() or
                     (coeff[i][-1] < 0).any()
                    ):
                likelihood = -np.inf
                return likelihood, coeff

            # calculate the likelihood of nore i
            L_for_node_i = self.get_node_likelihood(coeff[i], parent_list, i)

            # record the likelihood
            self.record[i][parent_list_str] = [L_for_node_i, coeff[i]]

            likelihood += L_for_node_i
        # BIC penalty
        likelihood -= np.count_nonzero(graph) * np.log(m) / 2

        return likelihood, coeff

    def Hill_Climb_search(self):
        n = self.n
        L = -np.inf
        best_edge_mat = np.zeros([n, n])
        step_mat = np.zeros([n, n])
        while True:
            stop_tag = True
            for new_edge_mat in tqdm(list(self.one_step_change_iterator(step_mat))):

                new_L, coeff = self.get_total_likelihood(new_edge_mat)

                if new_L - L > 0:
                    L = new_L
                    stop_tag = False
                    best_edge_mat = new_edge_mat
            step_mat = best_edge_mat

            if stop_tag:
                print("HC_best_likelihood:", L)
                self.skeleton_mat = best_edge_mat
                return get_symmetric_matrix(self.skeleton_mat)

