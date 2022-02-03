from scipy.stats import pearsonr, spearmanr
import numpy as np
import math
import json
import argparse


NUM_BUCKETS = 20

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", type=str, required=True)
parser.add_argument("--f1_confidence_log",
                    type=str,
                    help="If provided, we'll write the model's f1 at various levels of confidence")
parser.add_argument("--f1_coverage_log",
                    type=str,
                    help="If provided, we'll write the model's f1 at various levels of coverage")
args = parser.parse_args()

probs = []
log_probs = []
f1_scores = []

for line in open(args.predictions):
    datum = json.loads(line)
    log_prob = datum["model_scores"][0] if "model_scores" in datum else datum["normalized_logprobs"][0]
    log_probs.append(log_prob)
    probs.append(math.exp(log_prob))
    f1_scores.append(datum["f1_scores"] if "f1_scores" in datum else datum["f1"][0])

print(f"Pearson correlation between prob and F1 score: {pearsonr(probs, f1_scores)}")
print(f"Pearson correlation between log prob and F1 score: {pearsonr(log_probs, f1_scores)}")
print(f"Spearman correlation between prob and F1 score: {spearmanr(probs, f1_scores)}")
print(f"Spearman correlation between log prob and F1 score: {spearmanr(log_probs, f1_scores)}")

## ECE computation

bucket_size = 1 / NUM_BUCKETS
bucket_limits = [bucket_size]
while bucket_limits[-1] <= 1.0:
    bucket_limits.append(bucket_limits[-1] + bucket_size)

buckets = {x: [] for x in bucket_limits}

for prob, score in sorted(zip(probs, f1_scores), key=lambda x: x[0]):
    for limit in bucket_limits:
        if limit >= prob:
            buckets[limit].append((prob, score))
            break

errors = []
num_points = []
for limit in bucket_limits:
    num_points.append(len(buckets[limit]))
    bucket_score = (sum([x[1] for x in buckets[limit]]) / len(buckets[limit])) if buckets[limit] else 0.0
    bucket_confidence = (sum([x[0] for x in buckets[limit]]) / len(buckets[limit])) if buckets[limit] else 0.0
    errors.append(abs(bucket_score - bucket_confidence))

ece = sum([x * y for x, y in zip(num_points, errors)]) / sum(num_points)
print(f"ECE: {ece}")


mean = lambda x: sum(x) / len(x)

confidence_thresholded_f1s = []
for threshold in np.arange(min(probs), max(probs), 0.01):
    bucket = [f1 for prob, f1 in zip(probs, f1_scores) if prob >= threshold]
    thresholded_f1 = mean(bucket)
    confidence_thresholded_f1s.append((threshold, thresholded_f1))
    if round(threshold, 2) in [0.25, 0.50, 0.75, 0.99]:
        print(f"Confidence threshold: {threshold}, F1 score: {thresholded_f1}, Prediction ratio: {len(bucket) / len(probs)}")

if args.f1_confidence_log:
    with open(args.f1_confidence_log, "w") as outfile:
        for threshold, f1 in confidence_thresholded_f1s:
            print(f"{round(threshold, 2)}\t{f1}", file=outfile)

if args.f1_coverage_log:
    _, f1_scores = zip(*sorted(zip(probs, f1_scores), key=lambda x: x[0], reverse=True))
    with open(args.f1_coverage_log, "w") as outfile:
        for ratio in np.arange(0.01, 1.01, 0.01):
            bucket = f1_scores[:int(ratio * len(f1_scores))]
            print(f"{round(ratio, 2)}\t{mean(bucket)}", file=outfile)
