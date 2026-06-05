import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics

from transformers import SegformerForSemanticSegmentation

class AerialSegmentationModel(L.LightningModule):
    def __init__(
            self,
            model_name: str,
            num_classes: int,
            ):
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)

        self.train_jaccard = torchmetrics.MulticlassJaccardIndex(num_classes=self.num_classes)
        self.val_jaccard = torchmetrics.MulticlassJaccardIndex(num_classes=self.num_classes)

    def forward(self, inputs):
        logits = self.model(inputs).logits
        logits = F.interpolate(logits, size=inputs.shape[2:], mode='bilinear', align_corners=False)

        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["images"])
        labels = batch["masks"]

        CE_loss = F.cross_entropy(logits, labels)
        self.log("train/loss", CE_loss, on_epoch=True, prog_bar=True)

        train_jaccard = self.train_jaccard(logits.argmax(dim=1), labels)
        self.log("train/jaccard", train_jaccard, on_epoch=True, prog_bar=True)

        return CE_loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["images"])
        labels = batch["masks"]

        CE_loss = F.cross_entropy(logits, labels)
        self.log("val/loss", CE_loss, on_epoch=True, prog_bar=True)

        val_jaccard = self.val_jaccard(logits.argmax(dim=1), labels)
        self.log("val/jaccard", val_jaccard, on_epoch=True, prog_bar=True)

        return CE_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {"params": self.model.segformer.parameters(), "lr": 1e-5},
            {"params": self.model.decode_head.parameters(), "lr": 1e-4}
        ])

        return optimizer
