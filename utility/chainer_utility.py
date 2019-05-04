from pathlib import Path

import chainer
from tb_chainer import SummaryWriter


class TensorBoardReport(chainer.training.Extension):
    def __init__(self, writer=None, isOnGPU=True):
        self.writer = writer
        self.isOnGPU = isOnGPU

    def __call__(self, trainer: chainer.training.Trainer):
        if self.writer is None:
            self.writer = SummaryWriter(Path(trainer.out))

        observations = trainer.observation
        n_iter = trainer.updater.iteration
        for n, v in observations.items():
            if isinstance(v, chainer.Variable):
                v = v.data
            if self.isOnGPU and isinstance(v, chainer.cuda.cupy.ndarray):
                v = chainer.cuda.to_cpu(v)

            self.writer.add_scalar(n, v, n_iter)
