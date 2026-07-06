from pathlib import Path
import torch
import json
from torch.utils.data import DataLoader
from dataset.yolo_detection_dataset import YoloDetectionDataset, detection_collate
from evaluation.detr_eval import build_model as build_detr_model, postprocess_detr_outputs, CLASS_NAMES as DETR_CLASSES
from evaluation.efficientdet_eval import build_model as build_effdet_model, detections_to_predictions, CLASS_NAMES as EFFDET_CLASSES
from dataset.efficientdet_dataset import KittiEfficientDetAdaptor, EfficientDetDataset, get_valid_transforms


def debug_detr():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = YoloDetectionDataset(Path('data/processed'), 'val', DETR_CLASSES, input_size=(800, 800))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=detection_collate)
    model = build_detr_model(num_classes=len(DETR_CLASSES))
    model.load_state_dict(torch.load(Path('results/detr/detr_kitti.pt'), map_location=device))
    model.to(device)
    model.eval()
    images, targets = next(iter(loader))
    images = [img.to(device) for img in images]
    image_sizes = [(img.shape[1], img.shape[2]) for img in images]
    outputs = model(pixel_values=torch.stack(images))
    dets = postprocess_detr_outputs(model, outputs, image_sizes, score_threshold=0.25)
    print('DETR batch predictions')
    for i, det in enumerate(dets):
        print(f' image {i} preds {len(det)}')
        print(det[:10])
        print(' gt boxes', targets[i]['boxes'][:5])
        print(' gt labels', targets[i]['labels'][:5])
        break


def debug_effdet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adaptor = KittiEfficientDetAdaptor(data_root=Path('data/processed'), split='val', class_names=EFFDET_CLASSES)
    dataset = EfficientDetDataset(adaptor, transforms=get_valid_transforms(512))
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=False, collate_fn=lambda batch: ([item[0] for item in batch], [item[1] for item in batch], [item[2] for item in batch]))
    model = build_effdet_model('tf_efficientdet_d0', num_classes=len(EFFDET_CLASSES))
    state_dict = torch.load(Path('results/efficientdet/efficientdet_kitti.pt'), map_location=device)
    from evaluation.efficientdet_eval import align_state_dict_prefix
    state_dict = align_state_dict_prefix(state_dict, model, prefix='model.')
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    images, targets, image_ids = next(iter(loader))
    images = images.to(device)
    detections = model(images)
    print('EffDet raw output type', type(detections))
    if isinstance(detections, tuple):
        detections = detections[0]
    print('EffDet dim', detections.shape)
    for i, det in enumerate(detections):
        preds = detections_to_predictions(det, score_threshold=0.25)
        print(f' image {i} preds {len(preds)}')
        print(preds[:10])
        print(' gt boxes', targets[i]['bboxes'][:5])
        print(' gt labels', targets[i]['labels'][:5])
        break


if __name__ == '__main__':
    debug_detr()
    print('---')
    debug_effdet()
