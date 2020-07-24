import os

import torch
from progress.bar import Bar
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from SPAIR.SPAIR import SPAIR
from SPAIR.KLBuilder import KL_Builder

class Trainer:
    '''
    a class for train SPAIR.
    with support of drawing boundary, logging and scoring.
    '''
    def __init__(self, Implement=SPAIR, **config):
        save_cfg = self.checkpoint(config.get("model_path", None))
        if save_cfg:
            self.loadConfig(config, save_cfg["config"])
            self.start_epoch = save_cfg["epoch"]
            self.spair = Implement(**self.config.get('model', {}))
            self.spair.load_state_dict(save_cfg["model"])

        else:
            self.loadConfig(config)
            self.start_epoch = 0
            self.spair = Implement(**self.config.get('model', {}))
        
        self.kl_bulder = KL_Builder(**self.config['KL_Builder'])
        self.op = eval(self.optimizer)(self.spair.parameters(), **self.config.get(self.optimizer, {}))
        self.spair.to(self.device)

    @staticmethod
    def checkpoint(path)-> dict:
        if not path: return
        if os.path.exists(path):
            return torch.load(path)
        else:
            folder = os.path.dirname(path)
            if not os.path.exists(folder): os.makedirs(folder)  # make sure `save` will not raise exception

    def loadConfig(self, config, default: dict=None):
        if default: self.config = default.copy()
        else: self.config = {}
        self.config.update(config)
        self.summary = SummaryWriter(
            self.config["logdir"], 
            self.config["logdir"].split('/')[-1]
        )
        self.device = torch.device(self.config.get('device', 'cuda:0'))
        self.optimizer = self.config.get('optimizer', 'Adam')
        self._sup = self.config.get('KL_lambda', 1)

    def save(self):
        d = {
            "config": self.config, 
            "epoch": self.cur_epoch, 
            "model": self.spair.state_dict()
        }

        folder = os.path.dirname(self.config["model_path"])
        if not os.path.exists(folder): os.makedirs(folder)  # make sure `save` will not raise exception

        torch.save(d, self.config["model_path"])

    def loss(self, rec, target, param_dict: dict, global_step)-> torch.Tensor:
        recon_loss = binary_cross_entropy(rec, target, reduction='sum')
        norm_loss = self.kl_bulder.norm_KL(param_dict)
        kin_loss = self.kl_bulder.bin_KL(param_dict['pres'], global_step)
        assert not torch.isnan(recon_loss)
        self.summary.add_scalar('loss/reconstruct', recon_loss, global_step)
        self.summary.add_scalar('loss/normal', norm_loss, global_step)
        self.summary.add_scalar('loss/bernoulli', kin_loss, global_step)
        return recon_loss + self._sup * (norm_loss + kin_loss)

    def __histogram(self, tag: str, value: torch.Tensor, global_step=None, dim=None):
        with torch.no_grad():
            if dim:
                for i, v in enumerate(torch.split(value, 1, dim=dim)): 
                    self.summary.add_histogram("sample/%s/%s_%d" % (tag, tag, i), v, global_step)
            else:
                self.summary.add_histogram("sample/%s" % tag, value, global_step)

    def rectangle(self, rec, norm_box, pres, global_step):
        '''
        rec: [N, C, H_img, W_img]
        norm_box: [N, H*W, 4], y_center, x_center, height, width
        pres: [N, H*W, 1]
        '''
        with torch.no_grad():
            norm_box[:, :, :2] *= self.spair.encoder.image_size
            norm_box[:, :, 2:] *= self.spair.encoder.image_size / 2
            norm_box = torch.round_(norm_box)   # y_center, x_center, height / 2, width / 2
            norm_box = torch.stack((
                norm_box[:, :, 1] - norm_box[:, :, 3],  # xmin
                norm_box[:, :, 0] - norm_box[:, :, 2],  # ymin
                norm_box[:, :, 1] + norm_box[:, :, 3],  # xmax
                norm_box[:, :, 0] + norm_box[:, :, 2],  # ymax
            ), dim=-1)
            pres = torch.round_(pres).bool().squeeze_(-1)

            for i, (img, box, zpres) in enumerate(zip(rec, norm_box, pres)):
                box = torch.stack([b for b, z in zip(box, zpres) if z])
                self.summary.add_image_with_boxes('detect/rec_%d' % i, img, box, global_step)

    def reconstruct(self, X, bg=None):
        '''
        X: [N, H, W, C]
        return: [N, H, W, C]
        '''
        self.spair.eval()
        if X.dim() == 3: X = X.unsqueeze_(0)
        prev_device = X.device
        with torch.no_grad():
            return self.spair(X.to(self.device).permute(0, 3, 1, 2), bg.permute(0, 3, 1, 2)).to(prev_device)

    def train(self, X: torch.Tensor, bg=None):
        '''
        X shape: [N, H, W, C];
        '''
        # torch.autograd.set_detect_anomaly(True)
        if bg is None: bg = torch.zeros_like(X)
        data = TensorDataset(
            X.permute(0, 3, 1, 2).to(self.device),      # [N, 3, H, W]
            bg.permute(0, 3, 1, 2).to(self.device)      # [N, 3, H, W]
        )
        loader = DataLoader(data, **self.config.get("loader", {}))
        max_batch = len(loader)

        # ==============================
        # ========= train here =========
        def one_epoch():
            citer = max_batch * self.cur_epoch
            for batch, (R, B) in enumerate(loader):
                self.op.zero_grad()
                where, what, depth, pres, pd = self.spair.encoder(R)
                self.__histogram('where', pd['where'].mean, citer + batch, -1)
                self.__histogram('what', what, citer + batch)
                self.__histogram('depth', depth, citer + batch)
                self.__histogram('pres', pres, citer + batch)
                rec = self.spair.decoder(where, what, depth, pres, B)
                loss = self.loss(rec, R, pd, citer + batch)
                loss.backward()
                self.op.step()
                bar.next()

        # ========= train end ==========
        # ==============================

        for self.cur_epoch in range(self.start_epoch, self.config["max_epoch"]):
            bar = Bar('epoch%3d' % (self.cur_epoch + 1), max=max_batch)
            try: 
                self.spair.train()
                one_epoch()
                yield self.cur_epoch      # see what my caller wanna do after one epoch
                bar.finish()
            except KeyboardInterrupt:
                bar.finish()
                if 'Y' == input('save model? Y/n ').upper(): 
                    self.save()
                    print("Saved. Start from epoch%d next time." % (self.cur_epoch + 1))
                return

        self.cur_epoch = self.config["max_epoch"]
        self.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.summary.close()
