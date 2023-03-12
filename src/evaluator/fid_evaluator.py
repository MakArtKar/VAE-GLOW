# https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
from typing import Dict

import numpy as np
from scipy import linalg
from torch import Tensor
from torch.nn.functional import adaptive_avg_pool2d

from src.evaluator.base_evaluator import BaseEvaluator
from src.utils.fid.inception import InceptionV3


class FidEvaluator(BaseEvaluator):
    def __init__(self, dims, device):
        super().__init__()

        self.dims = dims
        self.device = device

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)
        self.model.eval()

        self.fake_acts = None
        self.real_acts = None
        self.reset()

    def reset(self):
        self.real_acts = []
        self.fake_acts = []

    def calc_acts(self, images: Tensor) -> np.ndarray:
        images = images.to(self.device)
        pred = self.model(images)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        return pred.squeeze(3).squeeze(2).cpu().numpy()

    def add_images(self, real_images: Tensor, fake_images: Tensor) -> None:
        self.real_acts.append(self.calc_acts(real_images))
        self.fake_acts.append(self.calc_acts(fake_images))

    @staticmethod
    def calculate_activation_statistics(acts):
        mu = np.mean(acts, axis=0)
        sigma = np.cov(acts, rowvar=False)
        return mu, sigma

    def calculate(self) -> Dict[str, float]:
        self.real_acts = np.concatenate(self.real_acts, axis=0)
        self.fake_acts = np.concatenate(self.real_acts, axis=0)

        m1, s1 = self.calculate_activation_statistics(self.real_acts)
        m2, s2 = self.calculate_activation_statistics(self.fake_acts)
        return {
            'fid': self.calculate_frechet_distance(m1, s1, m2, s2)
        }

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
