
def cost_nbr_words(S, dataset):
    """get the cost of Summary which is the number of words
    Args:
      S: a list of senetences in summary
      V_counts: sparse matrix to represent all sentences in grams counts
    """
    return dataset.X_train_counts[S].sum()
