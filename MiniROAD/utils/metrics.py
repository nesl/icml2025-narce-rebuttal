# Evaluation code from LSTR

from multiprocessing import Pool
from collections import OrderedDict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def calibrated_average_precision_score(y_true, y_score):
    """Compute calibrated average precision (cAP), which is particularly
    proposed for the TVSeries dataset.
    """
    y_true_sorted = y_true[np.argsort(-y_score)]
    tp = y_true_sorted.astype(float)
    fp = np.abs(y_true_sorted.astype(float) - 1)
    tps = np.cumsum(tp)
    fps = np.cumsum(fp)
    ratio = np.sum(tp == 0) / np.sum(tp)
    cprec = tps / (tps + fps / (ratio + np.finfo(float).eps) + np.finfo(float).eps)
    cap = np.sum(cprec[tp == 1]) / np.sum(tp)
    return cap


def perframe_average_precision(prediction, ground_truth, class_names,
                               postprocessing=None, metrics='AP'):
    """Compute (frame-level) average precision between ground truth and
    predictions data frames.
    """
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)
        
    # Build metrics
    if metrics == 'AP':
        compute_score = average_precision_score
    elif metrics == 'cAP':
        # print('cAP')
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore backgroud class
    ignore_index = set([0])

    # Compute average precision
    result['per_class_AP'] = OrderedDict()
    result['num'] = OrderedDict()
    print(f"NUM FRAMES: {np.sum(ground_truth[:, 1:])}")
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            if np.any(ground_truth[:, idx]):
                ap_score = compute_score(ground_truth[:, idx], prediction[:, idx])
                result['per_class_AP'][class_name] = ap_score
                result['num'][class_name] = f'[true: {int(np.sum(ground_truth[:, idx]))}, pred:{int(np.sum(prediction[:,idx]))}, AP:{ap_score*100:.1f}]'
    result['mean_AP'] = np.mean(list(result['per_class_AP'].values()))

    return result

def get_stage_pred_scores(gt_targets, pred_scores, perc_s, perc_e):
    starts = []
    ends = []
    stage_gt_targets = []
    stage_pred_scores = []
    for i in range(len(gt_targets)):
        if gt_targets[i] == 0:
            stage_gt_targets.append(gt_targets[i])
            stage_pred_scores.append(pred_scores[i])
        else:
            if i == 0 or gt_targets[i - 1] == 0:
                starts.append(i)
            if i == len(gt_targets) - 1 or gt_targets[i + 1] == 0:
                ends.append(i)
    if len(starts) != len(ends):
        raise ValueError('starts and ends cannot pair!')

    action_lens = [ends[i] - starts[i] for i in range(len(starts))]
    stage_starts = [starts[i] + int(action_lens[i] * perc_s) for i in range(len(starts))]
    stage_ends = [max(stage_starts[i] + 1, starts[i] + int(action_lens[i] * perc_e)) for i in range(len(starts))]
    for i in range(len(starts)):
        stage_gt_targets.extend(gt_targets[stage_starts[i]: stage_ends[i]])
        stage_pred_scores.extend(pred_scores[stage_starts[i]: stage_ends[i]])
    return np.array(stage_gt_targets), np.array(stage_pred_scores)


def perstage_average_precision(prediction, ground_truth,
                               class_names, postprocessing, 
                               metrics='cAP'):
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)

    # Postprocessing
    if postprocessing is not None:
        ground_truth, prediction = postprocessing(ground_truth, prediction)

    # Build metrics
    if metrics == 'AP':
        compute_score = average_precision_score
    elif metrics == 'cAP':
        compute_score = calibrated_average_precision_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore backgroud class
    ignore_index = set([0])

    # Compute average precision
    for perc_s in range(10):
        perc_e = perc_s + 1
        stage_name = '{:2}%_{:3}%'.format(perc_s * 10, perc_e * 10)
        result[stage_name] = OrderedDict({'per_class_AP': OrderedDict()})
        for idx, class_name in enumerate(class_names):
            if idx not in ignore_index:
                stage_gt_targets, stage_pred_scores = get_stage_pred_scores(
                    (ground_truth[:, idx] == 1).astype(int),
                    prediction[:, idx],
                    perc_s / 10,
                    perc_e / 10,
                )
                result[stage_name]['per_class_AP'][class_name] = \
                    compute_score(stage_gt_targets, stage_pred_scores)
        result[stage_name]['mean_AP'] = \
            np.mean(list(result[stage_name]['per_class_AP'].values()))

    return result


def perframe_average_F1(prediction, ground_truth,
                               class_names, postprocessing, 
                               metrics='F1'):
    
    result = OrderedDict()
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)
    prediction = prediction.argmax(axis=-1)  # shape: (2000, 60)

    # # Postprocessing
    # if postprocessing is not None:
    #     ground_truth, prediction = postprocessing(ground_truth, prediction)
        
    # Build metrics
    if metrics == 'F1':
        compute_score = f1_score
    else:
        raise RuntimeError('Unknown metrics: {}'.format(metrics))

    # Ignore background class (usually class 0)
    ignore_index = set([0])

    # Prepare result containers
    result['per_class_F1'] = OrderedDict()
    result['num'] = OrderedDict()

    # Print number of frames (optional)
    print(f"NUM FRAMES: {ground_truth.shape[0] * ground_truth.shape[1]}")

    # Compute per-class F1
    for idx, class_name in enumerate(class_names):
        if idx not in ignore_index:
            # Create binary ground truth and prediction for class idx
            gt_binary = (ground_truth == idx).astype(int).reshape(-1)  # shape: (N*T,)
            pred_binary = (prediction == idx).astype(int).reshape(-1)  # shape: (N*T,)

            if np.any(gt_binary):  # if class is present in ground truth
                f1 = compute_score(gt_binary, pred_binary, zero_division=0)
                result['per_class_F1'][class_name] = f1
                result['num'][class_name] = f'[true: {int(np.sum(gt_binary))}, pred: {int(np.sum(pred_binary))}, F1: {f1 * 100:.1f}]'

    # Compute mean F1 over all classes
    result['mean_F1'] = np.mean(list(result['per_class_F1'].values()))
    return result