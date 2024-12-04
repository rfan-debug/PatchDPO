import os
import json
import argparse
from evaluation.evaluate_dreambooth_func import evaluate_scores


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser("metric", add_help=False)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--is_kosmosg", type=str2bool, default=False)
    return parser.parse_args()


args = parse_args()

clipt_score, clipi_score, dino_score = evaluate_scores(args.output_dir, args.data_root, is_kosmosg=args.is_kosmosg)
save_score_dict = {
    'clipt_score': str(clipt_score),
    'clipi_score': str(clipi_score),
    'dino_score': str(dino_score),
}
save_score_path = os.path.join(args.output_dir, 'all_score.json')
# Save the evaluation results
with open(save_score_path, 'w') as file:
    json.dump(save_score_dict, file, indent=4)
print('clipt_score: %.4f, clipi_score: %.4f, dino_score: %.4f' % (clipt_score, clipi_score, dino_score))