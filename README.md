# GiveMeSomeCredit

1. Tell us how you validate your model, which, and why you chose such evaluation technique(s).

I used a 5-fold cross-validation method. The advantage of this method is that you do not lose any training data in the process in contrast with a classical train_test_split technique. It also increases your confidence in the performance of the model. Indeed, the AUC is computed by averaging five AUC scores on five different testing set.

The disadvantage of this technique is that it raises the possibility of you overfitting the training data.

2. What is AUC? Why do you think AUC was used as the evaluation metric for such a problem? What are other metrics that you think would also be suitable for this competition?

AUC - ROC curve is a performance measurement for the classification problems at various threshold settings. The ROC curve is a plot of the True Positive Rate (TPR) against the False Positive Rate (FPR). The TPR is on the y-axis and the FPR is on the x-axis.

AUC stands for Area under the Curve. The curve in question is the ROC curve. The area is the integral below the ROC curve. Its maximum value is 1 and the minimum value is 0. When the AUC is 1, TPR is equal to 1 and FPR is equal to 0. This means that the model makes perfect predictions.

The advantage of the AUC - ROC curve is that it evaluates the model for each probability threshold that the model outputs.

An alternative evaluation metric could have been to plot a precision and recall curve based on the F1-score. Usually, this metric is better suited for unbalanced datasets in terms of their dependent variable y.

3.  What insight(s) do you have from your model? What is your preliminary analysis of the given dataset?

It is hard to extract intuition from our model and this dataset. The most performing models are gradient boosting trees which perform 60% better than neural networks. These ensemble classifiers aggregate weak learners sequentially. Each new tree is added to minimise the overall loss. These models usually work better when there is no clear causal relationship between the dependent and independent variables. This is validated by the correlation matrix that I computed. There is no clear positive and negative correlation between y and X. Moreover, I plotted each variable by the dependent variable (SeriousDlqin2yrs). We do not notice any clear linear separation between the dependent variables and y.

Secondly, looking at the feature importance scores of different classifiers, each boosted tree use radically different variables to separate the dataset. For example,  NumberOfTimes90DaysLate, NumberOfTime30-59DaysPastDueNotWorse and NumberOfTime60-89DaysPastDueNotWorse represent over 50% of the feature contribution for each tree of LightGBM. On the other hand, for XGBoost, the split is more equally distributed over all variables. For the latter model, Depthratio, age and MonthlyIncome are the most important features. This diversity of important variables in building trees depending on the algorithm used shows that the features available are not good predictors of y.

4. Can you get into the top 100 of the private leaderboard, or even higher?

No. I have an average AUC of 0.864708 on the training cross-validation. My best submission on the Kaggle challenge had a performance of 0.86050 (Public Score) and 0.86671 (Private Score). The difference between the training and testing performance. The model generalises quite well.