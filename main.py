from el_credit_scorer import ELCreditScorer
from nn_credit_scorer import NNCreditScorer
import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    if args.model == "EL":
        cs = ELCreditScorer()
        cs.evaluate_classifiers()
        cs.hyperparameter_tuning()
        cs.funetuning_model()
        cs.submit_kaggle(args.submission_path)
        cs.plot_feature_importance()
    elif args.model == "NN":
        model = NNCreditScorer(
            args.path_training,
            args.path_test,
            device,
            args.epochs,
            args.learning_rate,
            args.batch_size,
        )
        model.train()
        model.submit_kaggle(args.submission_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Credit Scoring Algorithm for Kaggle Competition GiveMeSomeCredit")
    parser.add_argument("--model", choices=["NN", "EL"], default="EL",
                        help='EL: ensemble learning classifiers. NN: Neural Network for binary classification.')
    parser.add_argument("--submission_path", type=str,
                        default="./data/submission.csv")
    parser.add_argument("--path_training", type=str,
                        default="./data/cs-training.csv")
    parser.add_argument("--path_test", type=str, default="./data/cs-test.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)
