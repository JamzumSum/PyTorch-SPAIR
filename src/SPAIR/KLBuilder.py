from itertools import product

import torch
from torch.distributions import Normal, Uniform, kl_divergence
from torch.nn.functional import binary_cross_entropy
from math import log

# 默认 VAE 先验
DefaultPrior = {
    'where': [
        [0., 0., -2., -2.], 
        [1., 1., 0.5, 0.5]
    ],
    'depth': [0., 1.],
    'what': [0., 1.],
}

def safe_log(t):
    return torch.log(t + 1e-9)

class KL_Builder:

    def __init__(
        self, prior, 
        s_start=1000000.0, s_end=0.0125, 
        decay_rate=0.1, decay_step=1000., 
        staircase=False, log_space=False
    ):
        self.prios = DefaultPrior
        self.prios.update(prior)
        self.s_start = s_start
        self.s_end = s_end
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.staircase = staircase
        self.log_space = log_space

    def exponential_decay(self, step):
        '''
        A decay helper function for computing decay of
        :param global_step:
        :param start:
        :param end:
        :param decay_rate:
        :param decay_step:
        :param staircase:
        :param log_space:
        :return:
        '''
        if self.staircase:
            t = step // self.decay_step  # 整除
        else:
            t = step / self.decay_step
        value = (self.s_start - self.s_end) * (self.decay_rate ** t) + self.s_end

        if self.log_space: value = log(value + 1e-6)
        return value

    def bin_KL(self, z_pres, step):
        # -----z_pres_loss------

        # obj_log_odds = torch.clamp(z_pres, -10., 10.)
        # obj_log_odds
        # # 为存在的对象添加相关噪声
        # eps = 10e-10
        # u = Uniform(0, 1)
        # u = u.rsample(obj_log_odds.size())  # 取样
        # noise = torch.log(u + eps) - torch.log(1.0 - u + eps)  # 生成噪声
        # obj_pre_sigmoid = (obj_log_odds + noise) / 1.0
        # obj_prob = torch.sigmoid(obj_pre_sigmoid)  # Object Classification
        # z_pres_prob = obj_prob

        N, H, W, _ = z_pres.shape
        HW = H * W
        device =z_pres.device
        z_pres_prob = z_pres

        count_support = torch.arange(HW + 1, dtype=torch.float32).to(device)  # 对应论文中的C
        # FIXME starts at 1 output and gets small gradually
        # 对参数s进行指数衰减
        count_prior_log_odds = self.exponential_decay(step)
        count_prior_log_odds = torch.FloatTensor([count_prior_log_odds]).to(device)
        count_prior_prob = 1 / ((-count_prior_log_odds).exp() + 1)  # 1 - s

        # p(z_pres|C=nz(z_pres)) 带参数s的几何分布
        count_distribution = (1 - count_prior_prob) * (count_prior_prob ** count_support)

        count_distribution = count_distribution / count_distribution.sum()  # 归一化
        count_distribution = count_distribution.repeat(N, 1)  # (N, H*W+1)

        # number of symbols discovered so far
        count_so_far = torch.zeros(N, 1, device=device)        # (N, 1)
        i = 0
        z_pres_loss = 0.0

        for h, w in product(range(H), range(W)):
            p_z_given_Cz = torch.clamp(count_support - count_so_far, min=0., max=(HW - i)) / (HW - i)

            # Adds a new dim to to each vector for dot product [N, H*W + 1, ?]
            _count_distribution = count_distribution[:, None, :]
            _p_z_given_Cz = p_z_given_Cz[:, :, None]
            # 计算先验p(zpres)
            # equivalent of doing batch dot product on two vectors
            p_z = torch.bmm(_count_distribution, _p_z_given_Cz).squeeze(-1)  #三维tensor的矩阵乘法

            prob = z_pres_prob[:, h, w]
            prob = prob.to(device)
            # TODO This is for testing uniform dist

            #  计算prob和p_z的KL散度,用safe_log来防止Nan的情况
            _obj_kl = (
                    prob * (safe_log(prob) - safe_log(p_z))
                    + (1 - prob) * (safe_log(1 - prob) - safe_log(1 - p_z))
            )
            z_pres_loss += torch.mean(torch.sum(_obj_kl, dim=[1]))

            # Check if object presents (0.5 threshold)
            # original: tf.to_float(tensors["obj"][:, h, w, b, :] > 0.5), but obj should already be rounded
            sample = torch.round(z_pres[1, h, w])
            # Bernoulli prob
            mult = sample * p_z_given_Cz + (1 - sample) * (1 - p_z_given_Cz)

            # update count distribution
            count_distribution1 = mult * count_distribution
            normalizer = count_distribution1.sum(dim=1, keepdim=True).clamp(min=1e-6)
            count_distribution = count_distribution1 / normalizer
            count_so_far += sample
            i += 1
        assert not torch.isnan(z_pres_loss)
        return z_pres_loss

    def norm_KL(self, attribute: dict):
        '''
        Compute KL loss of all attributes.
        :param dict including mean and std of z_pres, z_where, z_what, z_depth
        :return: KL loss including z_pres, z_where, z_what and z_depth
        '''
        norm_kl = 0.0
        z_pres = attribute['pres']
        device = z_pres.device
        from collections.abc import Iterable
        kl_priors = {}
        for z_name, (mean, std) in DefaultPrior.items():
            assert (min(std) if isinstance(std, Iterable) else std) > 0

            dist = Normal(
                torch.FloatTensor(mean).to(device) if isinstance(mean, Iterable) else mean, 
                torch.FloatTensor(std).to(device) if isinstance(std, Iterable) else std
            )
            kl_priors[z_name] = dist

        for attr_name, value in attribute.items():
            if attr_name in kl_priors:
                loss = kl_divergence(value, kl_priors[attr_name]) * z_pres
                loss = torch.sum(loss, dim=tuple(range(1, loss.dim()))).mean()
                # print(attr_name + "_loss: %s" % loss.item())
                assert not torch.isnan(loss.detach()).max()
                assert not torch.isinf(loss.detach()).max()
                norm_kl += loss

        return norm_kl
