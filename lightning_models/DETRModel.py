from transformers import DetrForObjectDetection, ConditionalDetrForObjectDetection, YolosForObjectDetection
from HintDETR import DetrForObjectDetectionH
import pytorch_lightning as pl
import torch

processors = {
    "yolos" : "hustvl/yolos-tiny",
    "detr": "facebook/detr-resnet-50",
    "cdetr": "microsoft/conditional-detr-resnet-50"
}

tagsWithSize = {
    "color_pos_size": 0,
    "color_pos": 1,
    "color_size": 2,
    "pos_size": 3,
    "color": 4,
    "pos": 5,
    "size": 6,
    "N/A": 7
    # "good": 8
}

tags = {
    "color_pos": 0,
    "color": 1,
    "pos": 2,
    "N/A": 3
}


tags2names = {v: k for k, v in tags.items()}


class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, train_dataloader, val_dataloader, checkp=None, config=None):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        if not (checkp or config):
            self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                                num_labels=len(tags),
                                                                ignore_mismatched_sizes=True)
        elif not config:
            self.model = DetrForObjectDetection.from_pretrained(checkp)
        else:
            self.model = DetrForObjectDetection(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.val_dl = val_dataloader
        self.train_dl = train_dataloader

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs
    
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                weight_decay=self.weight_decay)
        
        return optimizer

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

class ConditionalDetr(Detr):
    def __init__(self, lr, lr_backbone, weight_decay, train_dataloader, val_dataloader, checkp=None, config=None):
        pl.LightningModule.__init__(self)
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        if not (checkp or config):
            self.model = ConditionalDetrForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50",
                                                                num_labels=len(tags),
                                                                ignore_mismatched_sizes=True)
        elif not config:
            self.model = ConditionalDetrForObjectDetection.from_pretrained(checkp)
        else:
            self.model = ConditionalDetrForObjectDetection(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.val_dl = val_dataloader
        self.train_dl = train_dataloader

class Yolos(pl.LightningModule):
    def __init__(self, lr, weight_decay, train_dataloader, val_dataloader, checkp=None, config=None):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        if not (checkp or config):
            self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny",
                                                                num_labels=len(tags) - 1,
                                                                ignore_mismatched_sizes=True)
        elif not config:
            self.model = YolosForObjectDetection.from_pretrained(checkp)
        else:
            self.model = YolosForObjectDetection(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_dl = val_dataloader
        self.train_dl = train_dataloader

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values)

        return outputs
    
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        
        return optimizer

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

class DetrH(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, train_dataloader, val_dataloader, checkp=None, config=None):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        if not (checkp or config):
            self.model = DetrForObjectDetectionH.from_pretrained("facebook/detr-resnet-50",
                                                                revision="no_timm", 
                                                                num_labels=len(tags) - 1,
                                                                ignore_mismatched_sizes=True)
        elif not config:
            self.model = DetrForObjectDetectionH.from_pretrained(checkp)
        else:
            self.model = DetrForObjectDetectionH(config)
        
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.val_dl = val_dataloader
        self.train_dl = train_dataloader

    def forward(self, pixel_values, pixel_mask, hint_position):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, hint_position=hint_position)

        return outputs
    
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        hint_position = batch['boxes']
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels, hint_position=hint_position)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay, amsgrad=True)
        
        return optimizer

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl
