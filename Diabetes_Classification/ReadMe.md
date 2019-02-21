# Problem statement

Predict the diabetes status of a patient given their health measurements<br>
In previous efforts, I could not come up with a better accuracy so, I have tried to improve accuracy in this excercise.<br>
Additionally, I was more interested in setting a model that correctly predict those patients with the diabetes (true positive rate). Hence, I focus more sensitivity instead of specificity.<br>

# Dataset<br>

https://www.kaggle.com/uciml/pima-indians-diabetes-database/home

# Modules <br>
pandas, sklearn, numpy, seaborn, matplotlib<br>

# Conclusion<br>

When I initially ran model accuracy was 77.08% . Null accuracy is like a starting point and that was 67.7%. Right there, we have an improvement of 10%.This was case when our threshold was 0.5, sensitivity was 53% and specificity was 88%.<br>

We changed value of threshold to 0.3 as we wanted to predict those patients with the diabetes (true positive rate). This actually decreased our accuracy to 72.91%. We got a jump in recall-score( sensitivity) at 71% and specificity at 70.7%. A trade-off that reduced our accuracy. On other hand, we got an AUC of 82.9% .<br>

This outcome begs an important question how AUC and accuracy are different? Answer is: accuracy is based on one specific cutpoint, while ROC tries all of the cutpoint and plots the sensitivity and specificity. So when we compare the overall accuracy, we are comparing the accuracy based on some cutpoint. The overall accuracy varies from different cutpoint.
