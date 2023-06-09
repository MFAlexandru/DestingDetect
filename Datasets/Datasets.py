import torch
from PIL import Image, ImageEnhance
import torchvision
import json
import os
from lightning_models.DETRModel import tags

def collate_fn(processor, batch):
    pixel_values = [item[0] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    batch['boxes'] = torch.stack([l['boxes'].squeeze() if l['boxes'].squeeze().shape != (0, 4) else torch.tensor([0, 0, 0, 0]) for l in labels])

    return batch

class TextStyleDataset(torchvision.datasets.VisionDataset):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "annotations_train.json" if train else "annotations_test.json")
        super().__init__(img_folder)
        with open(ann_file, "r") as ann:
            self.annotations = json.load(ann)

        # Exclude all categories with size
        self.ids = [f for f in list(sorted(self.annotations.keys())) if "size" not in f]
        if not train:
            self.ids = self.ids[0:100]
        self.processor = processor
        self.train = train

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img = Image.open(os.path.join(self.root, "images", self.ids[idx] + ".png")).convert("RGB")

        target = self.annotations[self.ids[idx]]
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        annotations = [{"id": (100000 + idx) * 100 + idxx, "category_id": tags[t["tag"]],
                        "bbox": [t["found"]["x"], t["found"]["y"], t["found"]["width"], t["found"]["height"]],
                        "area": 0} for idxx, (_, t) in enumerate(target.items()) if t["tag"] != "good"]
        target = {'image_id': idx, 'annotations': annotations}

        # Enhance region
        if len(annotations) != 0:
            [x, y, w, h] = [int(f) for f in annotations[0]["bbox"]]
            # Enhance region
            region = img.crop((x, y, x + w, y + h))
            if len(set(region.getdata())) > 1:
                # Enhance the contrast of the region
                enhancer = ImageEnhance.Contrast(region)
                enhanced_region = enhancer.enhance(2.0)

                # Paste the enhanced region back to the image
                img.paste(enhanced_region, (x, y))

        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension
        return pixel_values, target

    def __len__(self) -> int:
        return len(self.ids)


class TextStyleDatasetContrast(TextStyleDataset):
    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        # feel free to add data augmentation here before passing them to the next step
        img = Image.open(os.path.join(self.root, "images", self.ids[idx] + ".png")).convert("RGB")

        target = self.annotations[self.ids[idx]]
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        annotations = [{"id": (100000 + idx) * 100 + idxx, "category_id": tags[t["tag"]],
                        "bbox": [t["found"]["x"], t["found"]["y"], t["found"]["width"], t["found"]["height"]],
                        "area": 0} for idxx, (_, t) in enumerate(target.items()) if t["tag"] != "good"]
        target = {'image_id': idx, 'annotations': annotations}

        # Enhance region
        if len(annotations) != 0:
            [x, y, w, h] = [int(f) for f in annotations[0]["bbox"]]
            # Enhance region
            region = img.crop((x, y, x + w, y + h))
            if len(set(region.getdata())) > 1:
                # Enhance the contrast of the region
                enhancer = ImageEnhance.Contrast(region)
                enhanced_region = enhancer.enhance(2.0)

                # Paste the enhanced region back to the image
                img.paste(enhanced_region, (x, y))

        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension
        return pixel_values, target