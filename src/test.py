import torch, json

from Trainer import Trainer
from Loader import load_SDNET2018

def main():
    with open('./misc/SPAIR.json') as f: config = json.load(f)

    X, bg = load_SDNET2018(True)
    N, H, W, _ = X.shape
    bg = bg.reshape(N, 1, 1, 3).repeat(1, H, W, 1)

    X, test = torch.split(X, [N - 1, 1])
    bg, testBg = torch.split(bg, [N - 1, 1])

    with Trainer(**config) as trainer:
        if trainer.start_epoch > 0:
            with torch.no_grad():
                trainer.spair.eval()
                trainer.summary.add_graph(trainer.spair, (
                    test.permute(0, 3, 1, 2).to(trainer.device), 
                    testBg.permute(0, 3, 1, 2).to(trainer.device)
                ))
        trainer.summary.add_images('reconstruct/origin', test, 0, dataformats='NHWC')
        for i in trainer.train(X, bg):
            rec = trainer.reconstruct(test, testBg) # [N, C, H, W]
            trainer.summary.add_images('reconstruct/reconstruct', rec, i)

if __name__ == "__main__":
    main()