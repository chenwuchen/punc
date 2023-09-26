from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from src.model.core import ZhprBert, LogLearningRateCallback
from src.dataset.dataset import PunctDataModule
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

if __name__ == "__main__":
    parser = ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)    
    parser.add_argument('--max_epochs', '-me', type=int, default=5)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', '-bs', type=int, default=8)
    parser.add_argument('--steps_per_epoch', '-spe', type=int, default=200000)

    args = parser.parse_args()
    model = ZhprBert(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath='log/mdl/',
        filename='epoch{epoch}_{step}',
        every_n_train_steps= 1001,
        save_on_train_epoch_end=True,
        save_top_k=3,
        save_weights_only=True, 
        monitor="train_loss", 
        mode='min', 
        auto_insert_metric_name=False,
    )

    logger = CSVLogger("./log/test/")
    trainer = Trainer(max_epochs=args.max_epochs, devices=1, logger=logger, default_root_dir="./log/test", callbacks=[checkpoint_callback, LogLearningRateCallback()])
    datamodule = PunctDataModule(batch_size=args.batch_size)

    trainer.fit(model, datamodule=datamodule)
