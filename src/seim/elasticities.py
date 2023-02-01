import numpy as np
import pandas as pd

import re
import logging

import matplotlib.pyplot as plt
from copy import deepcopy

# logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

class CalculateEffects:
    def __init__(self, weights_matrix, coefficients: dict[str, float], matrix_inverse=None):

        spatial_coef = dict(sorted({key : [value] for (key,value) in coefficients.items() if "rho_" in key}.items())) #Sort rho_ d o w
        logging.debug(f"{spatial_coef}")
        logging.debug("What")

        names = set([re.sub("ORIG_|INTRA_|DEST_", "", key) for key in coefficients.keys() if not re.search("rho|Intra|Intercept", key)])
        beta_o = [value for (key, value) in coefficients.items() if re.search("ORIG_", key)]
        beta_d = [value for (key, value) in coefficients.items() if re.search("DEST_", key)]
        beta_i = [value for (key, value) in coefficients.items() if re.search("INTRA_", key)]

        logging.info("Done betas")

        self.n: int = weights_matrix.shape[0]

        if matrix_inverse is None:
            w_d = np.kron(np.identity(weights_matrix.shape[0]), weights_matrix)
            w_o = np.kron(weights_matrix, np.identity(weights_matrix.shape[0]))
            w_w = np.kron(weights_matrix, weights_matrix)
            weights = [w_d, w_o, w_w]

            assert len(set([i.shape[0] for i in weights])) == 1, f"{set([i.shape[0] for i in weights])}"

            logging.info("Done weights")

            self.matrix_inverse = np.linalg.inv(np.identity(self.n**2) - sum([rho * w for (rho, w) in zip(spatial_coef.values(), weights)]))
            logging.debug(f"{spatial_coef.values()}")

        else:
            self.matrix_inverse = matrix_inverse


        # TODO: find some way to keep betas and names ordered (maybe ordered dict)

        logging.info("Starting te")
        
        self.te = CalculateEffects.calc_total_effects(names=names, beta_d=beta_d, beta_o=beta_o, beta_i=beta_i, n=self.n, matrix_inverse=self.matrix_inverse)

        logging.debug(f"TE:{self.te}")
        logging.debug(f"{beta_d}")
        logging.info("te done, starting de")

        self.de = CalculateEffects.calc_destination_effects(names=names, beta_d=beta_d, n=self.n, matrix_inverse=self.matrix_inverse)
        
        logging.debug(f"{self.de}")
        logging.debug(f"{beta_d}")
        logging.info("de done, starting oe")

        self.oe = CalculateEffects.calc_origin_effects(names=names, beta_o=beta_o, n=self.n, matrix_inverse=self.matrix_inverse)

        logging.debug(f"{self.oe}")
        logging.info("oe done, starting ie")

        self.ie = CalculateEffects.calc_intra_effects(names=names, beta_d=beta_d, beta_o=beta_o, beta_i=beta_i, n=self.n, matrix_inverse=self.matrix_inverse)

        logging.debug(f"{self.ie}")
        logging.info("ie done, starting ne")

        self.ne = CalculateEffects.calc_network_effects(names=names, beta_d=beta_d, beta_o=beta_o, beta_i=beta_i, n=self.n, matrix_inverse=self.matrix_inverse)
        # self.ne = {name: (te - de - oe - ie) for (name, te, de, oe, ie) in zip(self.te.keys(), self.te.values(), self.de.values(), self.oe.values(), self.ie.values())}

        logging.debug(f"{self.ne}")
        logging.info("__init__ done")

    def __str__(self) -> str:
        tab = "\t"
        nl = "\n"
        return f"Total effects:{(lambda x: f'{tab}{x[0]}:{x[1]}{nl}')(self.te.items())}"

    @staticmethod
    def _build_j_o(n: int):

        # Can build any J_{o_i}, i = {2, ..., n-1}
        lambda_j_o_i = lambda n, i: np.hstack((np.zeros((n, i-1)), np.ones((n, 1)), np.zeros((n, n-i))))

        # Make sure j_o_2 has shape n x n
        assert lambda_j_o_i(n, 2).shape == (n, n), f"{lambda_j_o_i(n, 2).shape}\n{lambda_j_o_i(n, 2)}"

        # Stacks (j_{o_1}, ..., j_{o_n}) vertically
        j_o = np.vstack((
            np.hstack((np.ones((n, 1)), np.zeros((n, n-1)))),
            *[lambda_j_o_i(n, i) for i in range(2, n)],
            np.hstack((np.zeros((n, n-1)), np.ones((n, 1))))
        ))

        assert j_o.shape == (n**2, n), f"{j_o.shape}\n{j_o}"

        return j_o
    
    @staticmethod
    def _build_j_d(n: int):

        # Can build any J_{d_i}, i = {2, ..., n-1}
        lambda_j_d_i = lambda n, i: np.vstack((np.zeros((i-1, n)), np.ones((1, n)), np.zeros((n-i, n))))

        # Make sure j_d_2 has shape n x n
        assert lambda_j_d_i(n, 2).shape == (n, n)

        # Stacks (j_{d_1}, ..., j_{d_n}) vertically
        j_d = np.vstack((
            np.vstack((np.ones((1, n)), np.zeros((n-1, n)))),
            *[lambda_j_d_i(n, i) for i in range(2, n)],
            np.vstack((np.zeros((n-1, n)), np.ones((1, n))))
        ))

        assert j_d.shape == (n**2, n), f"{j_d.shape}\n{j_d}"

        return j_d

    @staticmethod
    def _build_j_i(n: int):

        # Creates an all zero n x n matrix except for element (i, i) = 1
        def create_j_i_i(i):
            a = np.zeros((n, n))
            a[(i, i)] = 1
            return a
            
        j_i = np.vstack([create_j_i_i(i) for i in range(n)])

        assert j_i.shape == (n**2, n), f"{j_i.shape}\n{j_i}"

        return j_i

    # Function to be reused by all calc_... methods, work in progress
    # def _matrix_multiplication(n, j_matrices, names_and_betas, matrix_inverse):

    #     names, *betas = names_and_betas


    #     effects = {name :
    #         (1/n**2) * \
    #             np.matmul(np.ones((1, n**2)), \
    #                 np.matmul(
    #                     np.matmul(matrix_inverse, sum([j * beta for (j, beta) in zip(j_matrices, betas_r)])), 
    #                     np.ones((n, 1))
    #                 )
    #             )\
    #         for (name, betas_r) in zip(names, betas)}

    #     print(effects)

    #     return effects
        

    @staticmethod
    def calc_total_effects(names: set[str], beta_d: list[float], beta_o: list[float], beta_i: list[float],  n: int, matrix_inverse=None) -> dict[str, float]:
        
        
        if matrix_inverse is None:
            matrix_inverse = np.identity(n**2)

        j_d = CalculateEffects._build_j_d(n)
        j_o = CalculateEffects._build_j_o(n)
        j_i = CalculateEffects._build_j_i(n)

        # print(j_d)
        
        # total_effects = {name :
        #     (1/n**2) * \
        #         np.matmul(np.ones((1, n**2)), \
        #             np.matmul(
        #                 np.matmul(matrix_inverse, j_d * beta_d_r + j_o * beta_o_r + j_i * beta_i_r), 
        #                 np.ones((n, 1))
        #             )
        #         )\
        #     for (beta_d_r, beta_o_r, beta_i_r, name) in zip(beta_d, beta_o, beta_i, names)}

        # total_effects = {name :
        #     (1/n**2) * \
        #         np.matmul(np.ones((1, n**2)), \
        #             np.matmul(
        #                 np.matmul(matrix_inverse, j_d * beta_d_r + j_o * beta_o_r), 
        #                 np.ones((n, 1))
        #             )
        #         )\
        #     for (beta_d_r, beta_o_r, beta_i_r, name) in zip(beta_d, beta_o, beta_i, names)}

        total_effects = {name :
            (1/n**2) * \
                np.matmul(
                    np.matmul(
                        np.ones((1, n**2)),
                        np.matmul(matrix_inverse, j_d * beta_d_r + j_o * beta_o_r)
                    ),
                    np.ones((n, 1))
                )\
            for (beta_d_r, beta_o_r, beta_i_r, name) in zip(beta_d, beta_o, beta_i, names)}

        # total_effects = CalculateEffects._matrix_multiplication(n, (j_d, j_o), (names, beta_d, beta_o), matrix_inverse)
        
        return total_effects

    @staticmethod
    def calc_destination_effects(names: set[str], beta_d: list[float], n: int, matrix_inverse=None) -> dict[str, float]:

        if matrix_inverse is None:
            matrix_inverse = np.identity(n**2)

        j_d = CalculateEffects._build_j_d(n) - CalculateEffects._build_j_i(n)

        # print(j_d)

        destination_effects = {name :
            (1/n**2) * \
                np.matmul(
                    np.matmul(
                        np.ones((1, n**2)),
                        np.matmul(matrix_inverse, j_d * beta_d_r)
                    ),
                    np.ones((n, 1))
                )\
            for (beta_d_r, name) in zip(beta_d, names)}
        
        return destination_effects

    @staticmethod
    def calc_origin_effects(names: set[str], beta_o: list[float], n: int, matrix_inverse=None) -> dict[str, float]:
        
        if matrix_inverse is None:
            matrix_inverse = np.identity(n**2)

        j_o = CalculateEffects._build_j_o(n) - CalculateEffects._build_j_i(n)

        # print(j_o)
        
        # origin_effects = {name :
        #     (1/n**2) * \
        #         np.matmul(np.ones((1, n**2)), \
        #             np.matmul(
        #                 np.matmul(matrix_inverse, j_o * beta_o_r), 
        #                 np.ones((n, 1))
        #             )
        #         )\
        #     for (beta_o_r, name) in zip(beta_o, names)}

        origin_effects = {name :
            (1/n**2) * \
                np.matmul(
                    np.matmul(
                        np.ones((1, n**2)),
                        np.matmul(matrix_inverse, j_o * beta_o_r)
                    ),
                    np.ones((n, 1))
                )\
            for (beta_o_r, name) in zip(beta_o, names)}
            
        
        return origin_effects

    @staticmethod
    def calc_intra_effects(names: set[str], beta_d: list[float], beta_o: list[float], beta_i: list[float], n: int, matrix_inverse=None) -> dict[str, float]:
        if matrix_inverse is None:
            matrix_inverse = np.identity(n**2)

        j_i = CalculateEffects._build_j_i(n)

        # print(j_i)

        # […] the direct effect of a site attribute on an intra-regional flow corresponds to the sum of
        # three coefficients if the attribute is present in all three matrices DX, OX and IX 
        # - Dargel 2021
        
        # intra_effects = {name :
        #     (1/n**2) * \
        #         np.matmul(np.ones((1, n**2)), \
        #             np.matmul(
        #                 np.matmul(matrix_inverse, j_i * (beta_d_r + beta_o_r + beta_i_r)), 
        #                 np.ones((n, 1))
        #             )
        #         )\
        #     for (beta_d_r, beta_o_r, beta_i_r, name) in zip(beta_d, beta_o, beta_i, names)}

        intra_effects = {name :
            (1/n**2) * \
                np.matmul(
                    np.matmul(
                        np.ones((1, n**2)),
                        np.matmul(matrix_inverse, j_i * (beta_i_r))
                    ),
                    np.ones((n, 1))
                )\
            for (beta_d_r, beta_o_r, beta_i_r, name) in zip(beta_d, beta_o, beta_i, names)}
        
        return intra_effects

    @staticmethod
    def calc_network_effects(names: set[str], beta_d: list[float], beta_o: list[float], beta_i: list[float],  n: int, matrix_inverse=None) -> dict[str, float]:

        if matrix_inverse is None:
            matrix_inverse = np.identity(n**2)

        j_d = CalculateEffects._build_j_d(n)
        j_o = CalculateEffects._build_j_o(n)
        j_1 = np.ones((n**2, n))

        j_n = j_1 - j_d - j_o

        # network_effects = {name :
        #     (1/n**2) * \
        #         np.matmul(np.ones((1, n**2)), \
        #             np.matmul(
        #                 np.matmul(matrix_inverse, j_1 * (beta_d_r + beta_o_r + beta_i_r)), 
        #                 np.ones((n, 1))
        #             )
        #         )\
        #     for (beta_d_r, beta_o_r, beta_i_r, name) in zip(beta_d, beta_o, beta_i, names)}

        network_effects = {name :
            (1/n**2) * \
                np.matmul(
                    np.matmul(
                        np.ones((1, n**2)),
                        np.matmul(matrix_inverse, j_n * (beta_d_r + beta_o_r))
                    ),
                    np.ones((n, 1))
                )\
            for (beta_d_r, beta_o_r, beta_i_r, name) in zip(beta_d, beta_o, beta_i, names)}
                
        return network_effects