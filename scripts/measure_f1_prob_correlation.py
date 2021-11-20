from scipy.stats import pearsonr
import math
import json
import sys


probs = []
log_probs = []
f1_scores = []

for line in open(sys.argv[1]):
    datum = json.loads(line)
    log_probs.append(datum["model_scores"][0])
    probs.append(math.exp(datum["model_scores"][0]))
    f1_scores.append(datum["f1_scores"])

print(f"Correlation between prob and F1 score: {pearsonr(probs, f1_scores)}")
print(f"Correlation between log prob and F1 score: {pearsonr(log_probs, f1_scores)}")
