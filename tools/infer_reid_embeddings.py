import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import torchreid


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract ReID embeddings from images or tracklets'
    )
    parser.add_argument('input_path', type=str, help='Input image or directory path')
    parser.add_argument(
        '--weights', type=str, required=True, help='Path to checkpoint'
    )
    parser.add_argument(
        '--output', type=str, required=True, help='Output .pt file path'
    )
    parser.add_argument(
        '--model-name', type=str, default='osnet_ain_x1_0', help='Torchreid model name'
    )
    parser.add_argument('--device', type=str, default='cuda', help='cpu, cuda, cuda:0')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument(
        '--input-mode',
        type=str,
        default='tracklets',
        choices=['tracklets', 'images'],
        help='Interpret input_path as tracklet folders or a flat image set'
    )
    parser.add_argument('--height', type=int, default=256, help='Input image height')
    parser.add_argument('--width', type=int, default=128, help='Input image width')
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='L2-normalize embeddings before saving'
    )
    parser.add_argument(
        '--save-frame-embeddings',
        action='store_true',
        help='Save per-frame embeddings for tracklet mode'
    )
    return parser.parse_args()


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def collect_image_paths(root):
    root = Path(root)
    if root.is_file():
        if not is_image_file(root):
            raise ValueError(f'Unsupported image file: {root}')
        return [root]

    image_paths = sorted([p for p in root.iterdir() if is_image_file(p)])
    if not image_paths:
        raise ValueError(f'No images found in {root}')
    return image_paths


def collect_tracklets(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Tracklet input must be a directory: {root}')

    tracklets = []
    for subdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        frame_paths = sorted([p for p in subdir.iterdir() if is_image_file(p)])
        if frame_paths:
            tracklets.append((subdir.name, frame_paths))

    if not tracklets:
        raise ValueError(f'No tracklet subdirectories with images found in {root}')
    return tracklets


def batch_extract(extractor, paths, batch_size):
    embeddings = []
    for start in range(0, len(paths), batch_size):
        batch = [str(p) for p in paths[start:start + batch_size]]
        embeddings.append(extractor(batch).cpu())
    return torch.cat(embeddings, dim=0)


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = torchreid.utils.FeatureExtractor(
        model_name=args.model_name,
        model_path=args.weights,
        image_size=(args.height, args.width),
        device=args.device
    )

    payload = {
        'model_name': args.model_name,
        'weights': str(Path(args.weights)),
        'input_mode': args.input_mode,
        'input_path': str(input_path),
        'embeddings': {},
    }

    if args.input_mode == 'images':
        image_paths = collect_image_paths(input_path)
        image_embeddings = batch_extract(extractor, image_paths, args.batch_size)
        if args.normalize:
            image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

        payload['items'] = [str(path) for path in image_paths]
        payload['embeddings'] = image_embeddings
        print(f'Extracted embeddings for {len(image_paths)} images')

    else:
        tracklets = collect_tracklets(input_path)
        frame_embeddings_map = {}

        for tracklet_name, frame_paths in tracklets:
            frame_embeddings = batch_extract(extractor, frame_paths, args.batch_size)
            tracklet_embedding = frame_embeddings.mean(dim=0, keepdim=True)

            if args.normalize:
                frame_embeddings = F.normalize(frame_embeddings, p=2, dim=1)
                tracklet_embedding = F.normalize(tracklet_embedding, p=2, dim=1)

            payload['embeddings'][tracklet_name] = tracklet_embedding.squeeze(0)

            if args.save_frame_embeddings:
                frame_embeddings_map[tracklet_name] = {
                    'frames': [str(path) for path in frame_paths],
                    'embeddings': frame_embeddings
                }

            print(
                f'Processed tracklet {tracklet_name}: {len(frame_paths)} frames -> 1 embedding'
            )

        if args.save_frame_embeddings:
            payload['frame_embeddings'] = frame_embeddings_map

    torch.save(payload, output_path)
    print(f'Saved embeddings to {output_path}')


if __name__ == '__main__':
    main()
