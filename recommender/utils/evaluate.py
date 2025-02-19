import numpy as np
import matplotlib.pyplot as plt

# Evaluation
def calculate_map_at_n(predictions, ground_truth, top_n=12):
    """
    Calculate Mean Average Precision @ N (MAP@N)
    
    Parameters:
    predictions (list): a list of predicted article_ids.
    ground_truth (list): a list of actual article_ids (ground truth).
    top_n (int): Number of top predictions to consider (default is 12).
    
    Returns:
    float: MAP@N score.
    """
    def precision_at_k(k, predicted, actual):
        if len(predicted) > k:
            predicted = predicted[:k]
        return len(set(predicted) & set(actual)) / k

    def average_precision(predicted, actual):
        if not actual:
            return 0.0
        ap = 0.0
        relevant_items = 0
        for k in range(1, min(len(predicted), top_n) + 1):
            if predicted[k-1] in actual:
                relevant_items += 1
                ap += precision_at_k(k, predicted, actual) * 1
        return ap / min(len(actual), top_n)
    
    return average_precision(predictions, ground_truth)


def compute_metrics(recommendations, test, top_n=12):
    purchase_dict = test.groupby('customer_id')['article_id'].agg(list)

    metrics = {
        "purchased": [],
        "hit_num": [],
        "precision": [],
        "recall": [],
        "recall_num": [],
        "map": []
    }
    for cid, purchased in purchase_dict.items():
        recommend_items = recommendations[cid][:top_n]

        hit = len(set(purchased).intersection(set(recommend_items)))
        precision = hit / len(recommend_items)
        recall = hit / len(purchased)
        purchased_num = len(purchased)
        
        metrics["purchased"].append(purchased_num)
        metrics["hit_num"].append(hit)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["recall_num"].append(len(set(recommend_items)))
        metrics["map"].append(calculate_map_at_n(list(recommend_items), list(purchased)))

    for k in metrics:
        metrics[k] = np.array(metrics[k]).mean()

    return metrics


def visualize_compare_recommendations(results, metrics, fig_width=24, vertical: bool = True):
    num_metrics = len(metrics)

    # Set a reasonable figure size dynamically
    # fig_width = 24  # Width suitable for one column of plots
    fig_height = num_metrics * 3  # Adjust height based on the number of metrics

    # Create the plots
    if vertical:
        fig, axes = plt.subplots(nrows=num_metrics, ncols=1, figsize=(fig_width, fig_height), sharey='row')
    else:
        fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=(fig_height, fig_width))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    if num_metrics == 1:
        axes = [axes]  # Ensure axes are iterable even for a single metric

    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # lime green
        '#17becf',  # cyan
        '#393b79',  # dark blue
        '#9c9ede'   # lavender
    ]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        rec_names = list(results.keys())
        values = [results[rec][metric] for rec in rec_names]
        ax.bar(rec_names, values, color=colors, alpha=0.8)
        ax.set_title(f'{metric.capitalize()}')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def collect_metrics(test_set, recommendations, top_n):
    results = {}
    for rec_name, recommendation in recommendations.items():
        results[rec_name] = compute_metrics(recommendation, test_set, top_n=top_n)
    return results


def compare_recommendations(test_set, recommendations, metrics, top_n, visualize: bool = True, fig_width=24, vertical: bool = True):
    # Get the data
    results = collect_metrics(test_set, recommendations, top_n=top_n)

    if visualize:
        visualize_compare_recommendations(results, metrics, fig_width, vertical)

    return results


def rank_calculate_mapk(actual, predicted, k=12):
    mapk = 0
    for user_id, group in predicted.groupby('customer_id'):
        actual_items = set(actual[user_id])
        pred_items = list(group['article_id'])
        score = 0
        hits = 0
        for i, p in enumerate(pred_items[:min(len(pred_items), k)]):
            if p in actual_items:
                hits += 1
                score += hits / (i + 1)
        mapk += score / min(len(actual_items), k)
    mapk /= len(predicted['customer_id'].unique())
    return mapk