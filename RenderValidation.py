import matplotlib.pyplot as plt
from transformers import DetrImageProcessor

from Datasets.Datasets import TextStyleDatasetContrast, collate_fn
from lightning_models.DETRModel import tags2names, Detr, processors
import torch
from PIL import Image
import os
from torch.utils.data import DataLoader

processor = DetrImageProcessor.from_pretrained(processors["detr"], max_size=900)


def collate(batch):
    return collate_fn(processor, batch)

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, scores, labels, boxes, imName):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    print(len(boxes.tolist()))
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        if label != len(tags2names) - 1:
            print(score, tags2names[label])
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            text = f'{tags2names[label]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig("results/" + imName + ".png")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_dataset = TextStyleDatasetContrast(img_folder="/home/malexandru/template24u_out_v2", processor=processor)
    val_dataset = TextStyleDatasetContrast(img_folder="/home/malexandru/template24u_out_v2", processor=processor, train=False)

    # model = DetrForObjectDetection.from_pretrained("/home/malexandru/checkPointA100_v4.4_fresh")

    train_dataloader = DataLoader(train_dataset, collate_fn=collate, batch_size=8, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate, batch_size=8, num_workers=8)

    # config = ConditionalDetrConfig(num_labels=len(tags), revision="no_timm", use_timm_backbone=False, num_queries=10)
    model = Detr.load_from_checkpoint("/home/malexandru/lightning_logs/Contrast_02/checkpoints/epoch=6-step=21136.ckpt", lr=1e-3, lr_backbone=1e-5, weight_decay=1e-4, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    # print(list(model.model.model.hint_position_layer.parameters()))

    for pixel_values, target in val_dataset:
        #print(target)
        pixel_values = pixel_values.unsqueeze(0).to(device)
        print(target)
        # boxes = target["boxes"]
        # boxes = boxes.to(device)
        with torch.no_grad():
        # forward pass to get class logits and bounding boxes
            outputs = model(pixel_values=pixel_values, pixel_mask=None)
        print("Outputs:", outputs.keys())

        # load image based on ID
        image_id = target['image_id'].item()
        image = val_dataset.ids[image_id]
        imageName = image
        print(imageName)
        image = Image.open(os.path.join(val_dataset.root, "images", image + ".png"))

        # postprocess model outputs
        width, height = image.size
        postprocessed_outputs = processor.post_process_object_detection(outputs,
                                                                        target_sizes=[(height, width)],
                                                                        threshold=0.3)
        results = postprocessed_outputs[0]
        cpuData = results['scores'].cpu()
        plot_results(image, results['scores'], results['labels'], results['boxes'], imageName)