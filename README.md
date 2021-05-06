# GiveMeSomeCredit

- Tell us how you validate your model, which, and why you chose such evaluation technique(s).
I used a 5-fold cross validation method. The advantage of this method is that you do not lose any training data in the process in contrast with a classical train_test_split technique.

The disadvantage of this technique is raises the possibility of you overfitting the training data.

- What is AUC? Why do you think AUC was used as the evaluation metric for such a problem? What are other metrics that you think would also be suitable for this competition?

AUC - ROC curve is a performance measurement for the classification problems at various threshold settings. The ROC curve is plotted with True Positive Rate (TPR) against the False Positive Rate (FPR) where TPR is on the y-axis and FPR is on the x-axis.

AUC stands for Area under the Curve. The curve in question is the ROC curve. The area is the integral below the ROC curve. Its maximum value is 1 and minimum value is 0. When the AUC is 1, TPR is equal to 1 and FPR is equal to 0. This means that the model makes perfect predictions.

The advantage of the AUC- ROC curve is that it evaluates the model for each probability threshold that the model outputs.

An alternative evaluation metric could have been to plot a precision and recall curve based on the F1-score. Usually, this metric is better suited for unbalanced datasets in terms of their dependent variable y.

- What insight(s) do you have from your model? What is your preliminary analysis of the given dataset?

By plotting the most importance features, we notice that 

- Can you get into the top 100 of the private leaderboard, or even higher?

No. I have a 