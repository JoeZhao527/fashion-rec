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
