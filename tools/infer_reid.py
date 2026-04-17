import argparse
import csv
from pathlib import Path

import torch

import torchreid


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run person ReID inference on query and gallery images'
    )
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Path to a query image or a directory of query images'
    )
    parser.add_argument(
        '--gallery',
        type=str,
        required=True,
        help='Path to a gallery directory of images'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='',
        help='Path to a trained checkpoint or pretrained ReID weights'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='osnet_x1_0',
        help='Model name recognized by torchreid.models.build_model'
    )
    parser.add_argument(
        '--height', type=int, default=256, help='Input image height'
    )
    parser.add_argument(
        '--width', type=int, default=128, help='Input image width'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Inference device, e.g. cpu, cuda, cuda:0'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='cosine',
        choices=['cosine', 'euclidean'],
        help='Distance metric for ranking'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=5,
        help='Number of gallery matches to show per query image'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Optional CSV output path for ranked matches'
    )
    return parser.parse_args()


def collect_images(path_str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f'Path does not exist: {path}')

    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f'Unsupported image file: {path}')
        return [path]

    images = sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f'No images found in directory: {path}')
    return images


def batch_extract(extractor, image_paths, batch_size=32):
    features = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = [str(p) for p in image_paths[start:start + batch_size]]
        batch_features = extractor(batch_paths).cpu()
        features.append(batch_features)
    return torch.cat(features, dim=0)


def main():
    args = parse_args()

    query_paths = collect_images(args.query)
    gallery_paths = collect_images(args.gallery)

    extractor = torchreid.utils.FeatureExtractor(
        model_name=args.model_name,
        model_path=args.weights,
        image_size=(args.height, args.width),
        device=args.device
    )

    query_features = batch_extract(extractor, query_paths)
    gallery_features = batch_extract(extractor, gallery_paths)

    distmat = torchreid.metrics.compute_distance_matrix(
        query_features, gallery_features, metric=args.metric
    )

    rows = []
    for query_index, query_path in enumerate(query_paths):
        distances = distmat[query_index]
        topk = min(args.topk, len(gallery_paths))
        ranking = torch.argsort(distances)[:topk]

        print(f'\nQuery: {query_path}')
        for rank, gallery_index in enumerate(ranking.tolist(), start=1):
            gallery_path = gallery_paths[gallery_index]
            score = float(distances[gallery_index].item())
            print(f'  Rank-{rank}: {gallery_path}  distance={score:.6f}')
            rows.append(
                {
                    'query': str(query_path),
                    'rank': rank,
                    'gallery': str(gallery_path),
                    'distance': score,
                }
            )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(
                csv_file, fieldnames=['query', 'rank', 'gallery', 'distance']
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved rankings to {output_path}')


if __name__ == '__main__':
    main()
