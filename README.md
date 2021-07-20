# Detection-of-Alzheimer's-disease


Alzheimer's disease is a progressive neurologic disorder that causes the brain to shrink (atrophy) and brain cells to die. Alzheimer's disease is the most common cause of dementia — a continuous decline in thinking, behavioral and social skills that affects a person's ability to function independently.

Symptoms: Amnesia; Dementia
Diseases or conditions caused: Dementia

![image](https://user-images.githubusercontent.com/51492488/126383049-7989bb80-7a5a-4414-859d-6709dbd44556.png)


Most of the cases of dementia observed in the age group of 70 - 80 years of Age.
Mens develop dementia at early age before 60 years while womens have tendency of dementia at later age of later than 60 years
In mens dementia starts at an education level of 4 years and most prevalent at education level of 12 years and 16 years and it can also extend upto more than 20 years of education level, while in womens dementia starts after 5 years of education level and most prevalent around 12 to 13 years of education level and it started to decrease as womens education level increase
Dementia is prevalent in Mens having highest and lowest socio economic status while womens having medium socio economic status have higher dementia cases.
Lower values of ASF close to 1 corresponds to severe dementia cases.
Severe dementia is diagnosed after minnimum 3 number of visits.



We conduct 10-fold cross-validation to figure out the best parameters for each model, SVM, Decision Tree, Random Forests, and AdaBoost. Since our performance metric is accuracy, we find the best tuning parameters by accuracy. In the end, we compare the accuracy, recall and AUC for each model.In case of medical diagnostics for non-life threatening terminal diseases like most neurodegenerative diseases it is important to have a high true positive rate so that all patients with alzheimer's are identified as early as possible. But we also want to make sure that the false positive rate is as low as possible since we do not want to misdiagnose a healthy adult as demented and begin medical therapy. Hence AUC seemed like a ideal choice for a performance measure.


Best accuracy on cross validation set is: 0.7445887445887445
Best parameter for c is:  1000
Best parameter for gamma is:  0.1
Best parameter for kernel is:  sigmoid
Test accuracy with the best parameters is 0.6944444444444444
Test recall with the best parameters is 0.8235294117647058
Test recall with the best parameter is 0.7012383900928792


