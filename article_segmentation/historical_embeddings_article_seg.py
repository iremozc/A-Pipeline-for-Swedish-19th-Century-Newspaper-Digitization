# -*- coding: utf-8 -*-
"""Article segmentation script

This script processes historical text using modern and historical word embeddings.
It performs segmentation and clustering of text.
"""

# Fix NumPy/Gensim compatibility issue
import os
import sys
import numpy as np
import importlib.util
import subprocess
import re

print("NumPy version:", np.__version__)

# Check if NumPy version is 2.x (incompatible with transformers)
numpy_version = np.__version__
USE_TRANSFORMERS = False # BERT is disabled in this version of the script
print("INFO: BERT (SentenceTransformer) embeddings are DISABLED for this script.")

if numpy_version.startswith('2.'):
    print("\nWARNING: NumPy 2.x detected. While BERT is disabled, this NumPy version might still have compatibility issues with other libraries.")
    print("Consider downgrading NumPy if you encounter unexpected errors.")
    # print("Options:")
    # print("1. Continue with fallback mode (no sentence transformer)") # Message adjusted as transformer is always off
    # print("2. Downgrade NumPy to 1.23.5 (compatible version)")
    # choice = input("Enter choice (1 or 2), or press Enter for option 1: ").strip() or "1"
    # 
    # if choice == "2":
    #     print("Downgrading NumPy to 1.23.5...")
    #     subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.23.5"], check=True)
    #     print("NumPy downgraded. Please restart the script.")
    #     sys.exit(0)
    
    # Use fallback mode
    # USE_TRANSFORMERS = False # This is now set unconditionally above
# else: # This else block is no longer needed as USE_TRANSFORMERS is always False
    # For NumPy 1.x, we can try to use transformers
    # USE_TRANSFORMERS = True


old_get_include = np.get_include
np.get_include = lambda: os.path.join(os.path.dirname(np.__file__), 'core', 'include')
sys.modules['numpy'] = np


import gensim
from gensim.models import KeyedVectors
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from collections import defaultdict
import math
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh

# Restore NumPy functions if needed
np.get_include = old_get_include

print("Gensim version:", gensim.__version__)

# Define paths - change these to your local paths
MODEL_PATH = ""
INPUT_FILE = ""
GT_FILE = ""
if len(sys.argv) >= 3:
    INPUT_FILE = sys.argv[1]
    GT_FILE = sys.argv[2]
elif len(sys.argv) == 2:
    INPUT_FILE = sys.argv[1]
    GT_FILE = None 
# Load historical word embeddings (Word2Vec format)
print(f"Loading historical model from {MODEL_PATH}")
try:
    hist_model = KeyedVectors.load(MODEL_PATH)
    print("Historical model loaded successfully")
except FileNotFoundError:
    print(f"Model file not found at {MODEL_PATH}")
    print("Please update the MODEL_PATH variable to the correct location")
    exit(1)

# Try to load SentenceTransformer only if we're not using the fallback mode
use_sentence_transformer = False # Will remain False as USE_TRANSFORMERS is False
# The following block for loading SentenceTransformer is removed as USE_TRANSFORMERS is False.
# if USE_TRANSFORMERS:
#     try:
#         from sentence_transformers import SentenceTransformer
#         print("Attempting to load the Swedish sentence transformer model...")
#         try:
#             model = SentenceTransformer("KBLab/sentence-bert-swedish-cased")
#             print("Sentence transformer model loaded successfully")
#             use_sentence_transformer = True
#         except Exception as e:
#             print(f"Error loading sentence transformer model: {e}")
#             # print("Falling back to Word2Vec only mode.")
#     except ImportError as e:
#         print(f"Could not import sentence_transformers: {e}")
#         # print("Falling back to Word2Vec only mode.")

# if not use_sentence_transformer:
#     print("Using Word2Vec for both modern and historical embeddings.")

# Read input file
print(f"Reading input file: {INPUT_FILE}")
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        ocr_lines = f.readlines()
    print(f"Successfully read {len(ocr_lines)} lines from input file")
except FileNotFoundError:
    print(f"Input file not found: {INPUT_FILE}")
    print("Please update the INPUT_FILE variable to the correct location")
    exit(1)
except UnicodeDecodeError:
    # Try alternative encoding if utf-8 fails
    try:
        with open(INPUT_FILE, 'r', encoding='latin-1') as f:
            ocr_lines = f.readlines()
        print(f"Successfully read {len(ocr_lines)} lines using latin-1 encoding")
    except Exception as e:
        print(f"Error reading input file: {e}")
        exit(1)

# Read ground truth file with original segment boundaries
print(f"Reading ground truth file: {GT_FILE}")
try:
    with open(GT_FILE, 'r', encoding='utf-8') as f:
        gt_lines = f.readlines()
    print(f"Successfully read {len(gt_lines)} lines from ground truth file")
except FileNotFoundError:
    print(f"Ground truth file not found: {GT_FILE}")
    print("Ground truth comparison will be skipped")
    gt_lines = None
except UnicodeDecodeError:
    # Try alternative encoding if utf-8 fails
    try:
        with open(GT_FILE, 'r', encoding='latin-1') as f:
            gt_lines = f.readlines()
        print(f"Successfully read {len(gt_lines)} lines using latin-1 encoding")
    except Exception as e:
        print(f"Error reading ground truth file: {e}")
        gt_lines = None

# Light pre-processing to ensure uniform data that are lowercased and stripped
def clean_line(line):
    # Remove leading/trailing whitespace
    line = line.strip()

    # Avoid lowercasing or stripping punctuation — let the BERT tokenizer handle that
    return line

ocr_lines = [clean_line(line) for line in ocr_lines if clean_line(line)]  # Skip empty lines

# Extract ground truth segments if available (marked by multiple empty lines)
gt_articles = []
if gt_lines:
    # Process the ground truth file to identify articles (separated by multiple empty lines)
    current_article = []
    empty_line_count = 0
    
    for line in gt_lines:
        clean = clean_line(line)
        if not clean:
            empty_line_count += 1
            if empty_line_count >= 3 and current_article:  # 3+ empty lines indicates an article boundary
                gt_articles.append(current_article)
                current_article = []
                empty_line_count = 0
        else:
            empty_line_count = 0
            current_article.append(clean)
    
    # Add the last article if it's not empty
    if current_article:
        gt_articles.append(current_article)
    
    print(f"Found {len(gt_articles)} articles in ground truth file")
    
    # Print sample of ground truth articles
    if gt_articles:
        print("\nSample of ground truth articles:")
        for i, article in enumerate(gt_articles[:3]):
            print(f"\nGround Truth Article {i+1} ({len(article)} lines):")
            for line in article[:3]:
                print(f"  {line}")
            if len(article) > 3:
                print(f"  ... (and {len(article)-3} more lines)")

# Print sample of the input data
print("\nSample of input data (first 5 lines):")
for i, line in enumerate(ocr_lines[:5]):
    print(f"{i+1}: {line}")


def evaluate_article_clustering(predicted_articles, gt_articles, ocr_lines):
    """
    Evaluate clustering quality when ground truth is at article level.
    
    Parameters:
    - predicted_articles: dict mapping cluster labels to lists of lines
    - gt_articles: list of lists, where each inner list contains lines from one article
    - ocr_lines: list of all preprocessed lines
    
    Returns:
    - dict with evaluation metrics
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Create article labels for each line
    line_to_pred_article = {}
    for article_id, lines in predicted_articles.items():
        for line in lines:
            # Find the index of this line in ocr_lines
            for i, ocr_line in enumerate(ocr_lines):
                if line == ocr_line:
                    line_to_pred_article[i] = article_id
                    break
    
    # Create ground truth article labels for each line
    line_to_gt_article = {}
    for gt_idx, article_lines in enumerate(gt_articles):
        for gt_line in article_lines:
            # Find the index of this line in ocr_lines
            for i, ocr_line in enumerate(ocr_lines):
                if gt_line == ocr_line:
                    line_to_gt_article[i] = gt_idx
                    break
    
    # Create label arrays for metrics calculation
    # Only include lines that appear in both GT and predicted
    common_lines = set(line_to_pred_article.keys()) & set(line_to_gt_article.keys())
    if not common_lines:
        return {
            "rand_index": 0.0,
            "nmi": 0.0,
            "coverage": 0.0,
            "purity": 0.0
        }
    
    y_true = [line_to_gt_article[i] for i in common_lines]
    y_pred = [line_to_pred_article[i] for i in common_lines]
    
    # Calculate standard clustering metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    
    # Calculate purity
    from collections import Counter
    purity_sum = 0
    for pred_cluster in set(y_pred):
        cluster_indices = [i for i, p in enumerate(y_pred) if p == pred_cluster]
        cluster_gt_labels = [y_true[i] for i in cluster_indices]
        most_common_gt = Counter(cluster_gt_labels).most_common(1)[0][1]
        purity_sum += most_common_gt
    
    purity = purity_sum / len(y_pred) if y_pred else 0
    
    # Calculate coverage (% of GT lines found in clusters)
    coverage = len(common_lines) / len(line_to_gt_article) if line_to_gt_article else 0
    
    return {
        "rand_index": ari,
        "nmi": nmi,
        "coverage": coverage,
        "purity": purity
    }


def get_embeddings(line):
    # Sentence-level embedding from BERT (384 or 768 dims depending on model)
    # sent_embed = model.encode(line)

    # Historical word-level embedding, these are static meaning they do not change based on the context of the sentence
    words = line.split()
    # Access vocab and vectors via hist_model.wv
    word_embeds = [hist_model.wv[word] for word in words if word in hist_model.wv.key_to_index]

    if word_embeds:
        hist_embed = np.mean(word_embeds, axis=0)
    else:
        # Use vector_size from hist_model.wv
        hist_embed = np.zeros(hist_model.wv.vector_size) # if oov, the value is zero

    return hist_embed

# Generate embeddings for all lines
print("\nGenerating embeddings for all lines...")
hist_embeddings = []

for line in ocr_lines:
    h_embed = get_embeddings(line)
    hist_embeddings.append(h_embed)

print(f"Generated embeddings for {len(hist_embeddings)} lines")

print("Historical embedding dimension:", hist_embeddings[0].shape)

# limiting the number of liunes to be represented
subset_hist = hist_embeddings[:200]

# Reduce dimensions to 2D
reducer = PCA(n_components=2)
reduced = reducer.fit_transform(subset_hist)

plt.figure(figsize=(10, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
plt.title("Historical Word Embeddings (Line-Level, Averaged)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.savefig("historical_embeddings_pca.png")

# # Fusion layer for combining embeddings
# class FusionLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim=256):
#         super(FusionLayer, self).__init__()
#         self.fusion = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
    
#     def forward(self, x):
#         return self.fusion(x)

# Non-parametric attention function
def non_parametric_self_attention(x): # for across lines
    # x: (1, seq_len, embed_dim)
    seq = x.squeeze(0)                   # → (seq_len, embed_dim)
    scores = torch.matmul(seq, seq.T)    # → (seq_len, seq_len)
    scores = scores / math.sqrt(seq.size(-1))
    attn = torch.softmax(scores, dim=-1) # → (seq_len, seq_len)
    out = torch.matmul(attn, seq)        # → (seq_len, embed_dim)
    return out.unsqueeze(0)              # → (1, seq_len, embed_dim)


if not hist_embeddings:
    print("No embeddings were generated. Exiting.")
    exit(1)

# Combine the embeddings
print("\nCombining embeddings...")
# fused_inputs = [np.concatenate([s, h]) for s, h in zip(sent_embeddings, hist_embeddings)]
# fused_tensor = torch.tensor(fused_inputs, dtype=torch.float32)
# Convert lists of numpy arrays to tensors
# Convert lists of numpy arrays to tensors
hist_tensor = torch.tensor(np.array(hist_embeddings), dtype=torch.float32)

# Get original dimensions
D_hist = hist_tensor.shape[1]

# Define common dimension for projection
common_dim = 256 # You can adjust this hyperparameter

# Define projection layers (these could be trained, but we'll keep them fixed like the attention)
proj_hist = nn.Linear(D_hist, common_dim)
# Freeze projection layers as well if you don't intend to train them
for p in proj_hist.parameters():
    p.requires_grad = False

# Project embeddings to the common dimension
print(f"Projecting embeddings to common dimension: {common_dim}")
with torch.no_grad():
    proj_hist_tensor = proj_hist(hist_tensor)
print("Applying concatenation for fusion...")
with torch.no_grad():
    # fused_outputs will have shape (num_lines, common_dim * 2)
    fused_outputs = proj_hist_tensor

print(f"Fused embeddings generated with shape: {fused_outputs.shape}")
contextualized_outputs = non_parametric_self_attention(fused_outputs.unsqueeze(0))[0]

print(f"Contextualized embeddings generated with shape: {contextualized_outputs.shape}")
# Calculate input dimension
# input_dim = fused_tensor.shape[1]
# print(f"Fusion input dimension: {input_dim}")

# Initialize and apply the fusion model
print("Applying fusion model...")





import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


fused_np = fused_outputs.detach().cpu().numpy()

fused_subset = fused_np[:200]

# Reduce dimensions
pca = PCA(n_components=2)
fused_2d = pca.fit_transform(fused_subset)

# Plot
plt.figure(figsize=(10, 7))
plt.scatter(fused_2d[:, 0], fused_2d[:, 1], alpha=0.7)
plt.title("Projected Historical Embeddings (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.savefig("fused_emebddings.png")
# Apply self-attention for contextualization
print("Applying self-attention for contextualization...")

# The FusionLayer section above is commented out, so fused_outputs is not defined.
# We will apply self-attention directly to fused_tensor (concatenated embeddings).
# Remove the line below that references the undefined fused_outputs:
# fused_tensor = fused_outputs.unsqueeze(0)  # REMOVE THIS LINE

# The non_parametric_self_attention function expects input shape (1, seq_len, embed_dim).
# Add the batch dimension when calling the function.
with torch.no_grad():
    # Add unsqueeze(0) here to create the batch dimension
    contextualized_outputs = non_parametric_self_attention(fused_outputs.unsqueeze(0))[0]

# Boundary detection
print("\nDetecting segment boundaries...")
contextualized_np = contextualized_outputs.detach().cpu().numpy()
# Compute pairwise cosine similarity between each line and the next
similarities = []
for i in range(len(contextualized_np) - 1):
    sim = cosine_similarity([contextualized_np[i]], [contextualized_np[i + 1]])[0, 0]
    similarities.append(sim)

def cuts_from_threshold(similarities, threshold):
    """Cut at every position where sim[i] < threshold."""
    return [i+1 for i, s in enumerate(similarities) if s < threshold]

def cuts_from_local_minima(similarities, window=2, top_k=None):
    """
    Find local minima in `similarities` over a sliding window of size `window`.
    If top_k is None, return *all* local minima; otherwise pick the top_k lowest minima.
    """
    minima = []
    N = len(similarities)
    for i in range(N):
        left  = max(0, i-window)
        right = min(N, i+window+1)
        if similarities[i] == min(similarities[left:right]):
            minima.append((i, similarities[i]))
    # minima holds (index, value).  We convert index→cut at i+1.
    if top_k:
        minima = sorted(minima, key=lambda x: x[1])[:top_k]
    return sorted([i+1 for i, _ in minima])

def smooth_similarities(similarities, window_size=3):
    """
    Smooth the similarity curve using a moving average to reduce noise.
    
    Parameters:
    - similarities: list of similarity scores
    - window_size: size of the moving average window (odd number recommended)
    
    Returns:
    - smoothed similarities
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    
    # Convert to numpy array for easier manipulation
    sim_array = np.array(similarities)
    
    # Apply Gaussian smoothing (sigma=1 is a moderate smoothing factor)
    smoothed = gaussian_filter1d(sim_array, sigma=1)
    
    # Ensure values stay in valid cosine similarity range [-1, 1]
    smoothed = np.clip(smoothed, -1.0, 1.0)
    
    return smoothed.tolist()


print("Using adaptive threshold for segmentation.")
# Define multiple threshold strategies to compare
thresholds = {
    "mean-std": np.mean(similarities) - np.std(similarities),
    "mean-0.5*std": np.mean(similarities) - 0.5 * np.std(similarities),
    "mean-1.5*std": np.mean(similarities) - 1.5 * np.std(similarities),
    "fixed-0.5": 0.5
}

# # Use the standard mean-std threshold by default
# threshold = thresholds["mean-std"]
# threshold_description = "mean-std"
# boundaries = cuts_from_threshold(similarities, threshold)

# print(f"Using {threshold_description} threshold method")
# print(f"Threshold value: {threshold:.4f}")
# print(f"Detected {len(boundaries)} boundaries at positions: {boundaries}")
# Use the standard mean-std threshold by default
threshold = thresholds["mean-1.5*std"]
threshold_description = "mean-1.5*std"
boundaries = cuts_from_threshold(similarities, threshold)

print(f"Using {threshold_description} threshold method")
print(f"Threshold value: {threshold:.4f}")
print(f"Detected {len(boundaries)} boundaries at positions: {boundaries}")

# Show how other thresholds would perform
for name, thresh in thresholds.items():
    if name != "mean-1.5*std":  # Skip the one we already used
        alt_boundaries = cuts_from_threshold(similarities, thresh)
        print(f"With threshold {name}: {len(alt_boundaries)} boundaries")

# Apply smoothing to reduce noise in similarity curve
print("\nApplying similarity curve smoothing...")
smoothed_similarities = smooth_similarities(similarities, window_size=3)

# Apply smoothed threshold detection
smoothed_threshold = np.mean(smoothed_similarities) - 1.5 * np.std(smoothed_similarities)
smoothed_boundaries = cuts_from_threshold(smoothed_similarities, smoothed_threshold)
print(f"After smoothing: detected {len(smoothed_boundaries)} boundaries with threshold {smoothed_threshold:.4f}")

# Apply local minima detection
print("\nApplying local minima detection...")
# Find all local minima
all_minima_boundaries = cuts_from_local_minima(similarities, window=2)
print(f"All local minima: detected {len(all_minima_boundaries)} boundaries")

# Find top-k local minima close to the number of expected segments from threshold method
expected_segments = len(boundaries) + 1
top_k_minima_boundaries = cuts_from_local_minima(similarities, window=2, top_k=max(1, expected_segments-1))
print(f"Top {expected_segments-1} local minima: detected {len(top_k_minima_boundaries)} boundaries")

# Find top-k local minima on smoothed similarities 
top_k_smoothed_minima_boundaries = cuts_from_local_minima(smoothed_similarities, window=2, top_k=max(1, expected_segments-1))
print(f"Top {expected_segments-1} local minima on smoothed curve: detected {len(top_k_smoothed_minima_boundaries)} boundaries")

# Let user choose which method to use
print("\nAvailable boundary detection methods:")
print(f"1. Threshold (mean-1.5*std): {len(boundaries)} boundaries")
print(f"2. Smoothed threshold: {len(smoothed_boundaries)} boundaries")
print(f"3. Local minima (all): {len(all_minima_boundaries)} boundaries")
print(f"4. Local minima (top-{expected_segments-1}): {len(top_k_minima_boundaries)} boundaries")
print(f"5. Smoothed local minima (top-{expected_segments-1}): {len(top_k_smoothed_minima_boundaries)} boundaries")


selected_method = 5  # Change this to select a different method
if selected_method == 1:
    chosen_boundaries = boundaries
    method_name = "threshold (mean-1.5*std)"
elif selected_method == 2:
    chosen_boundaries = smoothed_boundaries
    method_name = "smoothed threshold"
elif selected_method == 3:
    chosen_boundaries = all_minima_boundaries
    method_name = "all local minima"
elif selected_method == 4:
    chosen_boundaries = top_k_minima_boundaries
    method_name = f"top-{expected_segments-1} local minima"
else:  # selected_method == 5
    chosen_boundaries = top_k_smoothed_minima_boundaries
    method_name = f"top-{expected_segments-1} smoothed local minima"

print(f"\nUsing {method_name} method with {len(chosen_boundaries)} boundaries")
boundaries = chosen_boundaries  # Update boundaries with chosen method

# Create segments based on detected boundaries
segments = []
start = 0
for b in boundaries:
    segments.append((start, b))
    start = b
segments.append((start, len(contextualized_np)))  # Final segment

print(f"\nIdentified {len(segments)} segments:")
# Only print first 3 segments to avoid too much output
max_segments_to_show = min(3, len(segments))
for i, (start, end) in enumerate(segments[:max_segments_to_show]):
    print(f"\n--- Segment {i+1} (lines {start+1}-{end}) ---")
    # Print first few lines of each segment
    max_lines = min(5, end-start)
    for line in ocr_lines[start:start+max_lines]:
        print(line)
    if end-start > max_lines:
        print(f"... (and {end-start-max_lines} more lines)")

# Clustering
print("\nClustering segments into articles...")

from collections import defaultdict

# OLD APPROACH: Each segment is directly counted as an article
# n_found_clusters = len(segments)
# print(f"Found {n_found_clusters} articles")

# NEW APPROACH: Use SpectralClustering to group segments into articles
print("\nUsing SpectralClustering to group segments into articles...")

# Step 1: Calculate representative embedding for each segment
segment_embeddings = []
for start, end in segments:
    # Average the contextualized embeddings of all lines in this segment
    segment_embedding = np.mean(contextualized_np[start:end], axis=0)
    segment_embeddings.append(segment_embedding)

segment_embeddings = np.array(segment_embeddings)
print(f"Created embeddings for {len(segment_embeddings)} segments")

# Step 2: Determine the number of clusters (articles)
# Use the number from the threshold "mean-1.5*std"
threshold = np.mean(similarities) - 1.5 * np.std(similarities)
boundaries_for_clusters = cuts_from_threshold(similarities, threshold)
n_clusters = len(boundaries_for_clusters) + 1  # Number of segments = boundaries + 1
print(f"Detected {len(boundaries_for_clusters)} boundaries with mean-1.5*std threshold")
print(f"Setting number of clusters (articles) to {n_clusters}")

# Step 3: Compute pairwise similarity between segments
segment_similarity_matrix = cosine_similarity(segment_embeddings)
print(f"Created similarity matrix of shape {segment_similarity_matrix.shape}")

# Step 4: Apply SpectralClustering
if len(segment_embeddings) > 1:  # Make sure there's more than one segment
    spectral = SpectralClustering(
        n_clusters=min(n_clusters, len(segment_embeddings)-1),  # Can't have more clusters than segments
        affinity='precomputed',  # We're providing a precomputed similarity matrix
        random_state=42  # For reproducibility
    )
    
    # Fit the model to the segment similarity matrix
    cluster_labels = spectral.fit_predict(segment_similarity_matrix)
    
    # Step 5: Group segments by cluster
    segment_clusters = defaultdict(list)
    for segment_idx, cluster_label in enumerate(cluster_labels):
        segment_clusters[cluster_label].append(segment_idx)
    
    print(f"Grouped {len(segments)} segments into {len(segment_clusters)} articles using SpectralClustering")
    
    # Step 6: Create final articles by merging segments in the same cluster
    articles = []
    for cluster_label, segment_indices in segment_clusters.items():
        # Sort the segment indices to maintain the original order
        segment_indices.sort()
        
        # Find the continuous range of lines from the first to the last segment in this cluster
        first_segment_start = segments[segment_indices[0]][0]
        last_segment_end = segments[segment_indices[-1]][1]
        
        # Store article as a tuple of (start_line, end_line)
        articles.append((first_segment_start, last_segment_end))
    
    # Sort articles by start position
    articles.sort(key=lambda x: x[0])
    
    print("\nFinal Articles after clustering:")
    for i, (start, end) in enumerate(articles[:3]):  # Show first 3
        print(f"Article {i+1}: Lines {start+1}-{end} ({end-start} lines)")
    if len(articles) > 3:
        print(f"... and {len(articles)-3} more articles")
    
    n_found_clusters = len(articles)
    print(f"Found {n_found_clusters} articles after clustering (reduced from {len(segments)} initial segments)")
    
    # For further processing, replace 'segments' with 'articles'
    segments = articles
else:
    print("Only one segment detected, skipping clustering")
    n_found_clusters = len(segments)
    print(f"Found {n_found_clusters} articles")

# Detailed comparison with ground truth if available
if gt_articles and segments:
    print("\n=== DETAILED COMPARISON WITH GROUND TRUTH ===")
    print(f"Ground Truth: {len(gt_articles)} articles")
    print(f"Algorithm (after clustering): {len(segments)} segments")
    
    # Step 1: Length comparison
    gt_lengths = [len(article) for article in gt_articles]
    pred_lengths = [end-start for start, end in segments]
    
    print("\nSegment Length Comparison:")
    print(f"Ground Truth segment lengths: Min={min(gt_lengths)}, Max={max(gt_lengths)}, Avg={sum(gt_lengths)/len(gt_lengths):.1f}")
    print(f"Predicted segment lengths: Min={min(pred_lengths)}, Max={max(pred_lengths)}, Avg={sum(pred_lengths)/len(pred_lengths):.1f}")
    
    # Step 2: Create index-based mapping of GT segments
    gt_line_to_segment = {}
    for gt_idx, article in enumerate(gt_articles):
        for i, line in enumerate(article):
            gt_line_to_segment[i] = (gt_idx, i)
    
    # Step 3: Calculate segment-level matching
    # For each predicted segment, find best matching GT segment
    segment_matches = []
    for i, (start, end) in enumerate(segments):
        # Count which GT segment these lines belong to
        gt_counts = defaultdict(int)
        for j in range(start, end):
            if j in gt_line_to_segment:
                gt_counts[gt_line_to_segment[j][0]] += 1
        
        # Find best match
        if gt_counts:
            best_gt = max(gt_counts.items(), key=lambda x: x[1])
            gt_idx, count = best_gt
            total_pred = end - start
            total_gt = len([i for i, (gt_idx, _) in gt_line_to_segment.items() if gt_idx == gt_idx])
            precision = count / total_pred
            recall = count / total_gt
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            
            segment_matches.append({
                'pred_idx': i,
                'gt_idx': gt_idx,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'pred_size': total_pred,
                'gt_size': total_gt,
                'common_lines': count
            })
    
    # Sort matches by F1 score
    segment_matches.sort(key=lambda x: x['f1'], reverse=True)
    
    # Print top matches
    print("\nTop Segment Matches (Predicted → Ground Truth):")
    for i, match in enumerate(segment_matches[:5]):  # Show top 5
        print(f"Pred Segment {match['pred_idx']+1} (lines {segments[match['pred_idx']][0]+1}-{segments[match['pred_idx']][1]}) → GT Article {match['gt_idx']+1} (lines {match['gt_size']} lines)")
        print(f"  Precision: {match['precision']:.2f}, Recall: {match['recall']:.2f}, F1: {match['f1']:.2f}")
        print(f"  Common lines: {match['common_lines']}")
    
    # Calculate overall metrics
    avg_f1 = sum(m['f1'] for m in segment_matches) / len(segment_matches) if segment_matches else 0
    avg_precision = sum(m['precision'] for m in segment_matches) / len(segment_matches) if segment_matches else 0
    avg_recall = sum(m['recall'] for m in segment_matches) / len(segment_matches) if segment_matches else 0
    
    # Count perfectly matched segments (high F1)
    good_matches = sum(1 for m in segment_matches if m['f1'] > 0.8)
    
    print(f"\nOverall Segment Matching:")
    print(f"  Average Precision: {avg_precision:.2f}")
    print(f"  Average Recall: {avg_recall:.2f}")
    print(f"  Average F1: {avg_f1:.2f}")
    print(f"  Well-matched segments (F1>0.8): {good_matches} of {len(segment_matches)}")
    
    # Check if any GT articles were missed completely
    matched_gt_indices = set(m['gt_idx'] for m in segment_matches)
    missed_gt = set(range(len(gt_articles))) - matched_gt_indices
    if missed_gt:
        print(f"\nMissed Ground Truth Articles: {len(missed_gt)} articles not matched")
        print(f"  Article indices: {sorted(list(missed_gt))}")
    
    # Check if predicted segments got split/merged compared to GT
    from collections import Counter
    gt_match_counts = Counter(m['gt_idx'] for m in segment_matches)
    split_articles = [gt_idx for gt_idx, count in gt_match_counts.items() if count > 1]
    if split_articles:
        print(f"\nSplit Articles: {len(split_articles)} GT articles were split into multiple predicted segments")
        for gt_idx in split_articles[:3]:  # Show first 3
            pred_indices = [m['pred_idx'] for m in segment_matches if m['gt_idx'] == gt_idx]
            print(f"  GT Article {gt_idx+1} split into predicted segments: {[idx + 1 for idx in pred_indices]}")
boundary_window = 3
# Save results to file
if len(sys.argv) > 3:
    output_file = sys.argv[3]
else:
    output_file = os.path.splitext(INPUT_FILE)[0] + "_segments.txt"
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Segment clustering results for {INPUT_FILE}\n")
        f.write(f"Total lines: {len(ocr_lines)}\n")
        f.write(f"Total segments: {len(segments)}\n\n")
        
        # Add comparison with ground truth if available
        if gt_articles and segments:
            f.write(f"=== COMPARISON WITH GROUND TRUTH ===\n")
            f.write(f"Ground Truth: {len(gt_articles)} articles\n")
            f.write(f"Algorithm: {len(segments)} segments\n\n")
            
            f.write("Segment Matching Metrics:\n")
            f.write(f"  Average Precision: {avg_precision:.2f}\n")
            f.write(f"  Average Recall: {avg_recall:.2f}\n")
            f.write(f"  Average F1: {avg_f1:.2f}\n")
            f.write(f"  Well-matched segments (F1>0.8): {good_matches} of {len(segment_matches)}\n\n")
        
        for i, (start, end) in enumerate(segments):
            f.write(f"=== Segment {i+1} (lines {start+1}-{end}) ===\n")
            for line in ocr_lines[start:end]:
                f.write(line + '\n')
            f.write('\n')
    print(f"\nResults saved to {output_file}")
except Exception as e:
    print(f"Error saving results: {e}")

print("\nScript completed successfully")
# Additional Evaluation Metrics with Explanations
print("\n=== ENHANCED EVALUATION METRICS WITH INSIGHTS ===")

# Add WindowDiff and Pk metrics for boundary evaluation
from nltk.metrics.segmentation import windowdiff, pk
import numpy as np
from sklearn.metrics import v_measure_score, homogeneity_score, completeness_score
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score

# Only evaluate if ground truth is available
if gt_articles and segments:
    print("\n------ PART 1: SEGMENTATION EVALUATION (BOUNDARY DETECTION) ------")
    print("Evaluating how well the algorithm detects boundaries between articles")
    print("These metrics assess if segment boundaries are placed at the correct positions")
    
    # First, create reference and hypothesis segmentations as binary arrays
    # 1 marks a boundary, 0 marks no boundary
    ref_boundaries = np.zeros(len(ocr_lines))
    hyp_boundaries = np.zeros(len(ocr_lines))
    
    # Mark ground truth boundaries
    line_idx = 0
    for article in gt_articles:
        line_idx += len(article)
        if line_idx < len(ref_boundaries):
            ref_boundaries[line_idx] = 1
    
    # Mark predicted boundaries
    for boundary in boundaries:
        if boundary < len(hyp_boundaries):
            hyp_boundaries[boundary] = 1
    
    # Convert to strings for nltk metrics (1 = boundary, 0 = no boundary)
    ref_str = ''.join(str(int(x)) for x in ref_boundaries)
    hyp_str = ''.join(str(int(x)) for x in hyp_boundaries)
    
    # Calculate WindowDiff (lower is better, 0 is perfect)
    k = max(2, int(len(ref_str) / (2 * len(gt_articles))))
    wd = windowdiff(ref_str, hyp_str, k)
    
    # Calculate Pk (lower is better, 0 is perfect)
    p_k = pk(ref_str, hyp_str, k)
    
    print("\nSegmentation Boundary Evaluation Metrics:")
    print(f"  WindowDiff: {wd:.4f} (lower is better, 0 is perfect)")
    print(f"  Pk: {p_k:.4f} (lower is better, 0 is perfect)")
    
    # Interpret boundary evaluation metrics
    print("\nWhat These Segmentation Metrics Mean:")
    if wd < 0.3:
        print("  ✓ WindowDiff < 0.3: Boundary detection is good with few errors.")
    elif wd < 0.5:
        print("  ⚠ WindowDiff 0.3-0.5: Moderate boundary detection with some issues.")
        print("    - Your segmentation algorithm may be placing boundaries in the wrong locations")
        print("    - Try adjusting the boundary detection threshold or algorithm")
    else:
        print("  ✗ WindowDiff > 0.5: Poor boundary detection with significant errors.")
        print("    - Your segmentation algorithm is struggling to find article boundaries")
        print("    - Consider using different similarity metrics or thresholds")
        print("    - Check if threshold 'mean-std' is appropriate for your data")
    
    if p_k < 0.3:
        print("  ✓ Pk < 0.3: Good segmentation with few errors.")
    elif p_k < 0.5:
        print("  ⚠ Pk 0.3-0.5: Moderate segmentation with some issues.")
    else:
        print("  ✗ Pk > 0.5: Poor segmentation with many mistakes.")
    
    # Analyze boundary counts for additional insights
    if len(boundaries) > len(np.where(ref_boundaries == 1)[0]) * 1.5:
        print("  ⚠ Over-segmentation detected: Your segmentation algorithm is creating too many boundaries.")
        print("    - Try increasing the threshold to create fewer segments")
        print("    - Consider smoothing similarity scores to avoid noise")
    elif len(boundaries) * 1.5 < len(np.where(ref_boundaries == 1)[0]):
        print("  ⚠ Under-segmentation detected: Your segmentation algorithm is missing many boundaries.")
        print("    - Try decreasing the threshold to detect more segments")
        print("    - Consider using local minima detection instead of global threshold")
    
    # Measure exact boundary accuracy with window tolerance
    if len(boundaries) > 0 and len(ref_boundaries) > 0:
        # Calculate boundary hit rate with window tolerance
        tolerance = boundary_window  # Allow boundaries to be off by this many positions
        boundary_hits = 0
        ref_boundary_positions = np.where(ref_boundaries == 1)[0]
        
        for b in boundaries:
            # Check if this boundary is within tolerance of any reference boundary
            if any(abs(b - ref_b) <= tolerance for ref_b in ref_boundary_positions):
                boundary_hits += 1
        
        boundary_precision = boundary_hits / len(boundaries) if len(boundaries) > 0 else 0
        boundary_recall = boundary_hits / len(ref_boundary_positions) if len(ref_boundary_positions) > 0 else 0
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if boundary_precision + boundary_recall > 0 else 0
        
        print("\nSegmentation Boundary Detection Accuracy (with tolerance):")
        print(f"  Tolerance window: {tolerance} positions")
        print(f"  Boundary Precision: {boundary_precision:.4f} (higher is better, 1 is perfect)")
        print(f"  Boundary Recall: {boundary_recall:.4f} (higher is better, 1 is perfect)")
        print(f"  Boundary F1: {boundary_f1:.4f} (higher is better, 1 is perfect)")
        
        # Interpret boundary detection metrics
        print("\nWhat These Segmentation Boundary Metrics Mean:")
        if boundary_precision < 0.5:
            print("  ⚠ Low boundary precision: Many of your detected boundaries are incorrect.")
            print("    - Your segmentation algorithm is placing boundaries where they don't exist in ground truth")
            print("    - Try increasing the threshold to detect fewer, more confident boundaries")
        
        if boundary_recall < 0.5:
            print("  ⚠ Low boundary recall: Your segmentation algorithm is missing many true boundaries.")
            print("    - Many article transitions are not being detected")
            print("    - Try decreasing the threshold or using a more sensitive detection method")
        
        if boundary_f1 > 0.7:
            print("  ✓ Boundary F1 > 0.7: Good boundary detection performance.")
        elif boundary_f1 > 0.5:
            print("  ⚠ Boundary F1 0.5-0.7: Moderate boundary detection performance.")
        else:
            print("  ✗ Boundary F1 < 0.5: Poor boundary detection performance.")
            print("    - Your segmentation algorithm is struggling to identify article boundaries accurately")
            print("    - Consider a different approach to boundary detection")
    
    print("\n------ PART 2: SEGMENTATION EVALUATION (ARTICLE CLUSTERING) ------")
    print("Evaluating how well segments are grouped into coherent articles")
    print("These metrics assess if the clustering algorithm correctly groups segments into articles")
    
    # Create ground truth and prediction labels for each line
    gt_labels = np.zeros(len(ocr_lines), dtype=int)
    pred_labels = np.zeros(len(ocr_lines), dtype=int)
    
    # Assign article IDs to each line
    for gt_idx, article in enumerate(gt_articles):
        for line in article:
            for i, ocr_line in enumerate(ocr_lines):
                if line == ocr_line:
                    gt_labels[i] = gt_idx + 1  # +1 to avoid 0 as label
    
    for pred_idx, (start, end) in enumerate(segments):
        for i in range(start, end):
            if i < len(pred_labels):
                pred_labels[i] = pred_idx + 1  # +1 to avoid 0 as label
    
    # Calculate percentage of lines matched
    total_lines = len(ocr_lines)
    gt_matched_lines = np.sum(gt_labels > 0)
    pred_matched_lines = np.sum(pred_labels > 0)
    
    print(f"\nLine Matching Overview:")
    print(f"  Total lines in document: {total_lines}")
    print(f"  Lines matched in ground truth: {gt_matched_lines} ({gt_matched_lines/total_lines:.1%})")
    print(f"  Lines matched in prediction: {pred_matched_lines} ({pred_matched_lines/total_lines:.1%})")
    
    # Only consider lines that have both GT and prediction labels
    mask = (gt_labels > 0) & (pred_labels > 0)
    common_lines = np.sum(mask)
    print(f"  Lines matched in both: {common_lines} ({common_lines/total_lines:.1%})")
    
    if common_lines/total_lines < 0.5:
        print("  ⚠ Warning: Less than 50% of lines are matched in both ground truth and prediction.")
        print("    - This may indicate significant text preprocessing or alignment issues")
        print("    - Check your text cleaning and matching algorithms")
    
    if np.sum(mask) > 0:
        gt_labels_filtered = gt_labels[mask]
        pred_labels_filtered = pred_labels[mask]
        
        # Calculate V-measure, homogeneity, and completeness
        v_measure = v_measure_score(gt_labels_filtered, pred_labels_filtered)
        homogeneity = homogeneity_score(gt_labels_filtered, pred_labels_filtered)
        completeness = completeness_score(gt_labels_filtered, pred_labels_filtered)
        
        # Calculate adjusted mutual information
        ami = adjusted_mutual_info_score(gt_labels_filtered, pred_labels_filtered)
        
        # Calculate Fowlkes-Mallows index
        fm_index = fowlkes_mallows_score(gt_labels_filtered, pred_labels_filtered)
        
        print("\nArticle Clustering Evaluation Metrics:")
        print(f"  V-measure: {v_measure:.4f} (higher is better, 1 is perfect)")
        print(f"  Homogeneity: {homogeneity:.4f} (higher is better, 1 is perfect)")
        print(f"  Completeness: {completeness:.4f} (higher is better, 1 is perfect)")
        print(f"  Adjusted Mutual Information: {ami:.4f} (higher is better, 1 is perfect)")
        print(f"  Fowlkes-Mallows index: {fm_index:.4f} (higher is better, 1 is perfect)")
        
        # Interpret clustering metrics
        print("\nWhat These Article Clustering Metrics Mean:")
        if homogeneity > 0.7 and completeness < 0.5:
            print("  ⚠ High homogeneity but low completeness: Your clustering algorithm is over-segmenting articles.")
            print("    - Your predicted clusters are 'pure' but splitting actual articles")
            print("    - Try adjusting clustering parameters to produce fewer, larger clusters")
        elif homogeneity < 0.5 and completeness > 0.7:
            print("  ⚠ Low homogeneity but high completeness: Your clustering algorithm is under-segmenting articles.")
            print("    - Your predictions capture most of each article but mix multiple articles together")
            print("    - Try creating more clusters with smaller sizes")
        
        if v_measure > 0.7:
            print(f"  ✓ V-measure > 0.7: Good balance of homogeneity and completeness in your article clustering.")
        elif v_measure > 0.5:
            print(f"  ⚠ V-measure 0.5-0.7: Moderate article clustering quality.")
        else:
            print(f"  ✗ V-measure < 0.5: Poor article clustering quality.")
            print("    - Consider revisiting your embedding method and clustering approach")
            print("    - Try different clustering algorithms or similarity metrics")
        
        if ami > 0.5:
            print(f"  ✓ AMI > 0.5: Good cluster correlation with ground truth beyond chance.")
        else:
            print(f"  ⚠ AMI < 0.5: Weak correlation with ground truth, may be close to random clustering.")
            print("    - Your clustering isn't capturing the true article structure")
            print("    - Try different embeddings or fusion approaches")
        
        # B-Cubed Precision, Recall, and F1
        def b_cubed(gt_labels, pred_labels):
            """Calculate B-Cubed precision, recall, and F1 score."""
            precision_sum = 0
            recall_sum = 0
            n = len(gt_labels)
            
            # For each item i
            for i in range(n):
                # Find items with same prediction label as i
                pred_cluster_i = np.where(pred_labels == pred_labels[i])[0]
                # Find items with same ground truth label as i
                gt_cluster_i = np.where(gt_labels == gt_labels[i])[0]
                
                # Count items that are in both clusters
                correct = len(np.intersect1d(pred_cluster_i, gt_cluster_i))
                
                # Precision: proportion of items in pred_cluster_i that are in gt_cluster_i
                precision_i = correct / len(pred_cluster_i)
                precision_sum += precision_i
                
                # Recall: proportion of items in gt_cluster_i that are in pred_cluster_i
                recall_i = correct / len(gt_cluster_i)
                recall_sum += recall_i
            
            # Average precision and recall
            precision = precision_sum / n
            recall = recall_sum / n
            
            # F1 score
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return precision, recall, f1
        
        b_precision, b_recall, b_f1 = b_cubed(gt_labels_filtered, pred_labels_filtered)
        
        print("\nB-Cubed Metrics (line-level article assignment evaluation):")
        print(f"  B-Cubed Precision: {b_precision:.4f} (higher is better, 1 is perfect)")
        print(f"  B-Cubed Recall: {b_recall:.4f} (higher is better, 1 is perfect)")
        print(f"  B-Cubed F1: {b_f1:.4f} (higher is better, 1 is perfect)")
        import os
        csv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../b_cubed_scores-attention2stephist-mode5.csv')
        write_header = not os.path.exists(csv_file)
        with open(csv_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write("file,b_cubed_precision,b_cubed_recall,b_cubed_f1\n")
            f.write(f"{os.path.basename(INPUT_FILE)},{b_precision:.4f},{b_recall:.4f},{b_f1:.4f}\n")
        # Interpret B-Cubed metrics
        print("\nWhat These B-Cubed Metrics Mean:")
        if b_precision < 0.7:
            print("  ⚠ B-Cubed Precision < 0.7: Your article clustering is putting lines from different articles together.")
            print("    - Your clusters are mixing lines from different articles")
            print("    - Try improving the similarity calculation between lines")
        else:
            print("  ✓ B-Cubed Precision > 0.7: Lines you place in the same article generally belong together.")
            
        if b_recall < 0.7:
            print("  ⚠ B-Cubed Recall < 0.7: Your article clustering is separating lines that should be together.")
            print("    - Your algorithm is splitting articles across multiple clusters")
            print("    - Try producing larger, more cohesive clusters")
        else:
            print("  ✓ B-Cubed Recall > 0.7: Lines that belong in the same article are usually grouped together.")
        
        if b_f1 > 0.7:
            print("  ✓ B-Cubed F1 > 0.7: Good overall line-to-article assignment performance.")
        elif b_f1 > 0.5:
            print("  ⚠ B-Cubed F1 0.5-0.7: Moderate line-to-article assignment performance.")
        else:
            print("  ✗ B-Cubed F1 < 0.5: Poor line-to-article assignment performance.")
            print("    - The overall quality of assigning lines to the correct articles is low")
            print("    - Consider fundamental changes to your approach")
    
    # Overall Summary and Recommendations
    print("\n=== SUMMARY AND RECOMMENDATIONS ===")
    # Identify main issues
    segmentation_issues = []
    clustering_issues = []
    
    if wd > 0.3 or boundary_f1 < 0.7:
        segmentation_issues.append("boundary detection")
    
    if v_measure < 0.7:
        clustering_issues.append("article clustering quality")
    if b_f1 < 0.7:
        clustering_issues.append("line-to-article assignment")
    if common_lines/total_lines < 0.7:
        segmentation_issues.append("line matching/preprocessing")
    
    print("Areas needing improvement:")
    if segmentation_issues:
        print(f"  SEGMENTATION ISSUES: {', '.join(segmentation_issues)}")
    else:
        print("  SEGMENTATION: Good performance overall.")
        
    if clustering_issues:
        print(f"  CLUSTERING ISSUES: {', '.join(clustering_issues)}")
    else:
        print("  CLUSTERING: Good performance overall.")
        
    if not segmentation_issues and not clustering_issues:
        print("All metrics show good performance. Fine-tuning may still improve results.")
    
    # Specific recommendations based on metrics
    print("\nSpecific Recommendations:")
    
    if segmentation_issues:
        print("1. For better SEGMENTATION (identifying article boundaries):")
        if len(boundaries) > len(ref_boundary_positions) * 1.2:
            print("   - Reduce false boundaries by increasing the similarity threshold")
            print("   - Try using 'mean-1.5*std' threshold instead of 'mean-std'")
        elif len(boundaries) * 1.2 < len(ref_boundary_positions):
            print("   - Detect more boundaries by decreasing the similarity threshold")
            print("   - Try using 'mean-0.5*std' threshold instead of 'mean-std'")
        print("   - Experiment with local minima detection instead of global thresholding")
        print("   - Consider smoothing the similarity curve to reduce noise")
    
    if clustering_issues:
        print("2. For better CLUSTERING (grouping segments into articles):")
        if homogeneity < completeness:
            print("   - Your clusters contain mixed content from different ground truth articles")
            print("   - Try increasing the number of clusters")
            print("   - Experiment with different clustering algorithms (e.g., HDBSCAN)")
        else:
            print("   - Your clusters are splitting ground truth articles")
            print("   - Try decreasing the number of clusters")
        print("   - Improve embeddings by tuning the fusion approach or using different models")
    
    if common_lines/total_lines < 0.7:
        print("3. Improve text preprocessing and matching:")
        print("   - Review text cleaning to ensure better line matching between ground truth and OCR")
        print("   - Consider fuzzy matching for line identification")
        print("   - Check for encoding issues or inconsistent normalization")
    
else:
    print("Ground truth not available for enhanced evaluation metrics.")

# Add results to the output file
if gt_articles and segments:
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\n=== ENHANCED EVALUATION METRICS ===\n")
            f.write("------ PART 1: SEGMENTATION EVALUATION ------\n")
            f.write(f"WindowDiff: {wd:.4f} (lower is better, 0 is perfect)\n")
            f.write(f"Pk: {p_k:.4f} (lower is better, 0 is perfect)\n")
            f.write(f"Boundary F1 (tolerance {tolerance}): {boundary_f1:.4f} (higher is better, 1 is perfect)\n")
            f.write("\n------ PART 2: SEGMENTATION EVALUATION (ARTICLE CLUSTERING) ------\n")
            f.write(f"V-measure: {v_measure:.4f} (higher is better, 1 is perfect)\n")
            f.write(f"Homogeneity: {homogeneity:.4f} (higher is better, 1 is perfect)\n")
            f.write(f"Completeness: {completeness:.4f} (higher is better, 1 is perfect)\n")
            f.write(f"Adjusted Mutual Information: {ami:.4f} (higher is better, 1 is perfect)\n")
            f.write(f"Fowlkes-Mallows index: {fm_index:.4f} (higher is better, 1 is perfect)\n")
            f.write(f"B-Cubed Precision: {b_precision:.4f} (higher is better, 1 is perfect)\n")
            f.write(f"B-Cubed Recall: {b_recall:.4f} (higher is better, 1 is perfect)\n")
            f.write(f"B-Cubed F1: {b_f1:.4f} (higher is better, 1 is perfect)\n")
            
            # Write summary and recommendations
            f.write("\n=== SUMMARY AND RECOMMENDATIONS ===\n")
            if segmentation_issues:
                f.write(f"SEGMENTATION ISSUES: {', '.join(segmentation_issues)}\n")
            else:
                f.write("SEGMENTATION: Good performance overall.\n")
                
            if clustering_issues:
                f.write(f"CLUSTERING ISSUES: {', '.join(clustering_issues)}\n")
            else:
                f.write("CLUSTERING: Good performance overall.\n")
            
        print(f"Enhanced metrics with explanations saved to {output_file}")
    except Exception as e:
        print(f"Error saving enhanced metrics: {e}")
# === VISUALIZATION OF BOUNDARY DETECTION METHODS ===
plt.figure(figsize=(14, 6))

# Plot raw and smoothed similarities
plt.plot(similarities, label='Raw Similarities', alpha=0.5)
plt.plot(smoothed_similarities, label='Smoothed Similarities', linewidth=2)

# Plot threshold lines
plt.axhline(threshold, color='gray', linestyle='--', label='Raw Threshold')
plt.axhline(smoothed_threshold, color='blue', linestyle='--', label='Smoothed Threshold')

# Highlight boundary points
plt.scatter(all_minima_boundaries, [similarities[i-1] for i in all_minima_boundaries],
            marker='o', color='orange', label='All Local Minima')
plt.scatter(top_k_minima_boundaries, [similarities[i-1] for i in top_k_minima_boundaries],
            marker='D', color='red', label=f'Top-{expected_segments-1} Minima')
plt.scatter(top_k_smoothed_minima_boundaries, [smoothed_similarities[i-1] for i in top_k_smoothed_minima_boundaries],
            marker='X', color='green', label=f'Top-{expected_segments-1} Smoothed Minima')

# Format plot
plt.title("Similarity Curve and Detected Boundaries")
plt.xlabel("Line Index")
plt.ylabel("Cosine Similarity")
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

# Save or show the plot
plt.savefig("similarity_boundary_detection.png")
print("Boundary detection plot saved as 'similarity_boundary_detection.png'")
