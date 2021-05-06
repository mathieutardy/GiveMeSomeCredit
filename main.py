from credit_scorer import CreditScorer

if __name__ == "__main__":
    cs = CreditScorer()
    cs.evaluate_classifiers()
    cs.hyperparameter_tuning()
    cs.funetuning_model()
    cs.submit_kaggle("./data/submission.csv")
    cs.plot_feature_importance()
