import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

np.random.seed(42)
N = 33000
# Generate 19 clusters, some structure
true_labels = np.random.randint(0, 23, N)
pred_labels = true_labels.copy()
# Add noise to make it not perfect
noise = np.random.rand(N) < 0.3
pred_labels[noise] = np.random.randint(0, 19, noise.sum())

nmi = normalized_mutual_info_score(true_labels, pred_labels)
ami = adjusted_mutual_info_score(true_labels, pred_labels)

print(f"NMI = {nmi:.8f}")
print(f"AMI = {ami:.8f}")
print(f"Difference = {nmi - ami:.8f}")
