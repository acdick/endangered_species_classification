# Classification of Endangered Species

Predicting the Federal Listing Status of U.S. plant and animal species.

## U.S. Fish & Wildlife Service

Data for this project was collected from the U.S. Fish & Wildlife Service.

https://ecos.fws.gov

## Exploratory Data Analysis

### Species Group

* Dropped features that represented less than 1% of the population

![Species Group Distribution](/Plots/Species_Group_Distribution.png)

### State Distribution

* Dropped features that represented less than 1% of the population

![State Distribution](/Plots/State_Distribution.png)

### VIP Distribution

![VIP Distribution](/Plots/VIP_Distribution.png)

## Classification Models

* Dummy Classifier
* Logistic Regression
* K Nearest Neighbors
* Decision Tree
* Random Forest

### Baseline Model

![Class Imbalance](/Plots/Class_Imbalance.png)

![Baseline](/Plots/Baseline.png)

**Best Training Model by F1 Score**
* K Nearest Neighbors

![Baseline Train KNN](/Plots/Baseline_Train_KNN.png)

### Balanced Class Model with SMOTE Oversampling

![Class Balance](/Plots/Class_Balance.png)

![Balanced](/Plots/Balanced.png)

**Best Training Model by F1 Score**
* Decision Tree

![Balanced Train Decision Tree](/Plots/Balanced_Train_Decision_Tree.png)

### Tuned Hyper-Parameter and Balanced Class Model

![Balanced and Tuned](/Plots/Balanced_and_Tuned.png)

**Best Training Model by F1 Score**
* Logistic Regression

![Balanced and Tuned Train Logistic](/Plots/Tuned_Train_Logistic.png)

Most important features:

**States**
* Idaho
* Hawaii
* Wyoming

**Species Groups**
* Insects
* Crustaceans

![Feature Importance](/Plots/Feature_Importance.png)
