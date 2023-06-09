from transformers import DetrImageProcessor
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
import torch
from lightning_models.DETRModel import Detr, processors
from Datasets.Datasets import TextStyleDatasetContrast, collate_fn

processor = DetrImageProcessor.from_pretrained(processors["detr"], max_size=900)


def collate(batch):
    return collate_fn(processor, batch)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    train_dataset = TextStyleDatasetContrast(img_folder="/home/malexandru/template24u_out_v2", processor=processor)
    val_dataset = TextStyleDatasetContrast(img_folder="/home/malexandru/template24u_out_v2", processor=processor, train=False)

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    train_dataloader = DataLoader(train_dataset, collate_fn=collate, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate, batch_size=16, num_workers=4)
    batch = next(iter(train_dataloader))

    model = Detr.load_from_checkpoint("/home/malexandru/lightning_logs/version_7/checkpoints/epoch=2-step=7112.ckpt", lr=1e-5, lr_backbone=1e-5, weight_decay=1e-4, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    for p in model.model.model.parameters():
        p.requires_grad = True
    # for p in model.model.model.hint_position_layer.parameters():
    #     p.requires_grad = True
    # model.model.model.freeze_backbone()

    trainer = Trainer(max_steps=30000, gradient_clip_val=0.1, val_check_interval=100)
    trainer.fit(model)
