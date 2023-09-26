from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from src.model.core import BertPunc
from src.dataset.dataset  import PunctDataModule
from pytorch_lightning.loggers import CSVLogger


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)    
    parser.add_argument('--max_epochs', '-me', type=int, default=5)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-5)
    parser.add_argument('--batch_size', '-bs', type=int, default=8)
    parser.add_argument('--steps_per_epoch', '-spe', type=int, default=200000)
    parser.add_argument('--base_mdl', '-bm', type=str, default="mdl/bert-base-chinese")
    args = parser.parse_args()
    model = BertPunc(args)

    save_model = ModelCheckpoint(
        dirpath='output/mdl/',
        filename='epoch{epoch}_{step}',
        every_n_train_steps= 2501,
        save_on_train_epoch_end=True,
        save_top_k=5,
        save_weights_only=True, 
        monitor="train_loss", 
        mode='min', 
        auto_insert_metric_name=False,
    )


    early_stopping = EarlyStopping(
        'val_loss',
        mode='min',
        patience=5,
        check_on_train_epoch_end=False
    )

    logger = CSVLogger("./output", name="log")

    # trainer = Trainer.from_argparse_args(args, callbacks=[val_checkpoint_callback, early_stopping])
    trainer = Trainer(max_epochs=args.max_epochs, logger=logger, default_root_dir="./output", callbacks=[save_model])
    datamodule = PunctDataModule(batch_size=args.batch_size)

    # trainer.test(model,ckpt_path='lightning_logs/version_7/checkpoints/val-epoch=epoch=01-setp=step=6103-val_loss=val_loss=0.39.ckpt', datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    
    try:
        trainer.test(ckpt_path='best', datamodule=datamodule)
    except:
        print("best ckpt not found, use the last ckpt")
        trainer.test('last', datamodule=datamodule)
