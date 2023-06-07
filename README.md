# README

### Abstract

Proficiency in academic skills such as reading and numeracy, as well as educational attainment are correlated with individual economic outcomes and mortality. Given this correlation, it is imperative to identify potential causes for disparities in academic skills and educational attainment. Identifying potential causes serve as a starting point for interventions aimed at improving economic outlook and extending life expectancy.

This study aims to identify potential causes for disparities in educational attainment and achievement within Massachusetts Public Schools using data extracted from 2017 Massachusetts Department education reports. This data characterizes Massachusetts public elementary schools, middle schools, and high schools and provides academic proficiency data for all schools as well as educational attainment data for high schools.  The regression techniques employed are decision tree regression and random forest regression.

### Introduction

In 2020, the US Bureau of Labor Statistics reported that those without a high school diploma had an unemployment rate of 5.4% and median weekly earnings of $592. In contrast, those with a Bachelor's degree had an unemployment rate of 2.2% and median weekly earnings of $1248. Longitudinal studies, such as that conducted by Watts, have found measures of reading and mathematical skills to be correlated with career earnings. A 2011 report published by the US National Institute of Health found a 10-year disparity in life expectancy between those without a high school degree and those with a college degree.

Despite these correlations, levels of academic performance and educational attainment vary widely. In 2019, the U.S. Census Bureau reported that 13% of the population had no high school diploma, 48.5% had a high school diploma as their highest level of educational attainment, while 38.5% had a higher degree. Research has identified a range of individual factors contributing to these disparities including race, gender, and economic background.

The objective of this study is to identify potential contextual factors that contribute to disparities in academic performance and education attainment within Massachusetts schools. Contextual factors include characteristics of individual schools as well as characteristics of the location of the school. A related study carried out by Everson and Millsap (2004) looking at individual and school characteristics and SAT performance found that school characteristics directly influence SAT performance.

### Materials
    
Our primary data set is 1.54 MB in size and covers a total of 1800 middle schools and high schools for the year 2017. Our primary data set was downloaded from Kaggle (https://www.kaggle.com/ndalziel/massachusetts-public-schools-data).  In the primary data set, each school is characterized by location, the gender and racial composition of the student body, the percentage of students with disabilities and high needs as well as the percentage of students in economic distress. Schools are also characterized by the compensation offered to employees and the average expenditure per pupil.

Our primary data set is supplemented by American Survey Community Data for 2017, produced by the US Census Bureau (https://www.census.gov/programs-surveys/acs/data.html). This data describes locations in the US with populations over 5000 along social, economic, housing, and demographic dimensions. In our project, the subset describing Massachusetts locations will be used to characterize the towns in which schools are located.

For each high school in the primary dataset, indicators for educational attainment include graduation rates and the percentage of students pursuing various postsecondary accreditations. Reported measures of academic skills include average MCAS (Massachusetts Comprehensive Assessment System) scores for each grade, average SAT scores and the number of students taking AP classes along with the number of students attaining each grade. In total there are fourteen indicators for academic performance and educational attainment.
 
### Methods

To identify potential causes for disparity in academic achievement and attainment, we identified features that are important in performing Decision Tree Regression and Random Forest Regression for each output variable. These techniques were selected as they are less affected by collinearity than linear regression and are interpretable. Preprocessing was carried out using PySpark, plots were generated using MatplotLib and Seaborn, and regression was performed using the Sci Kit Learn API. 

During preprocessing, the Massachusetts public school data was used to generate 14 distinct data sets, each of which corresponds to a single output variable (indicator of academic achievement and attainment)). In each data set, only schools reporting on the pertinent output variable were included. Each data set was augmented using American Community Survey data. Features characterizing the town of each school were appended to each data point. Schools for which American Community Survey data was missing were removed from the datasets.

For each data set, the distribution of the output variable was plotted (https://bit.ly/3n9xXP3)  and correlations among features were calculated (https://bit.ly/3suxyYq). As output variables had a skewed distribution, stratified sampling was used to generate test and training data (https://bit.ly/3n9xXP3). Care was taken to ensure that schools from the same town were not present in both the test and training set. As the data sets were small, data instances with missing values were not removed; instead, mean imputation was used. Imputation was carried out separately on the training and test data sets and was only required for numerical data. Categorical features, including town, district, and grades taught at each school were encoded using one-hot-encoding.

Decision tree regression and random forest regression were performed initially with default parameters, and then grid search was used to tune hyperparameters. Regression was repeated using the best estimators from grid search. Decision tree structures for each decision tree regression were extracted as was the length of each decision tree (https://bit.ly/3eat7gd, https://bit.ly/2ROJM1v). The ten most important features for each type of regression for each output variable were identified (https://bit.ly/3x06AeJ). 

### Results

Three metrics, r2 (https://bit.ly/3eaxFTP), mean absolute error (https://bit.ly/3uZnfwV) and root mean square error (https://bit.ly/32qFuiK) were used to evaluate Decision Tree Regression and Random Forest Regression. 

As can be seen across all fourteen output variables, for all three metrics, hyperparameter tuning improves decision tree performance markedly. Hyperparameters were tuned by limiting decision tree length, imposing an upper limit on the number of leaf nodes and imposing a lower limit on the number of data instances in leaf nodes. Performance of particularly poorly performing baseline decision trees, for example those constructed for Percent of AP Exams receiving a score between 3-5,  3rd Grade MCAS Math CPI and 3rd Grade MCAS and English CPI are improved when the max number of leaf nodes are drastically limited (4, 5 11 respectively) and resulting decision trees are of a very short depth (2, 3, 6 respectively). These limits likely prevent overfitting, a very real challenge considering the small size of each data set. Hyperparameter tuning generally also improves performance in random forest regression, though by a smaller margin than for decision tree regression. This relatively small improvement in performance is likely due to overfitting being limited in baseline random forest regression due to a multiplicity of estimators.

In comparing baseline decision trees and baseline random forests, it can be seen that baseline random forest regression outperforms baseline decision tree regression for each output variable across all performance metrics. Interestingly, for two performance indicators, 10th Grade MCAS Math CPI and 5th Grade MCAS Math CPI, decision tree regression with tuned hyperparameters outperforms both baseline random forest regression and random forest regression with tuned hyperparameters.

In examining the most important features for the best performing models for both decision tree regression and random forest regression across output variables (https://bit.ly/3sxOwoC , https://bit.ly/3tuTm7n), it can be seen that although features derived from the American Survey Community Data do appear, they appear primarily as important features in decision tree regression, and their importance scores are relatively low. 

The most important features and those that appear most frequently in both decision tree regression and random forest regression are features that characterize schools. Of particular note, are the following features: percent of students who are economically disadvantaged, percent of students with high needs, student enrollment as well as features that characterize the racial and gender composition of the school. These features are frequently important for both decision tree regression and random forest regression for each output variable, and are frequently important across all output variables. 

In comparing the important features for decision tree regression and random forest regression for each output variable it can be seen that there is significant overlap particularly among the most important features.

### Discussion

The objective of this study was to identify possible contextual causes for disparities in educational achievement and attainment in Massachusetts schools in 2017. This was carried out by identifying the most important features for decision tree regression and random forest regression for 14 measures of academic achievement and attainment. Generally, the most important features were those that characterize schools including: percent of students who are economically disadvantaged, percent of students with high needs, enrollment numbers and racial and gender demographics of the school.

This study is quite limited, as only two types of regression were carried out. It is challenging to assess how well decision tree regression and random forest regression perform without a baseline, and thus how much weight can be placed in the important features identified. Furthermore, the size of each data set was quite small, in particular when this size is contrasted with the high dimensionality of the data. Therefore, it is unclear how well the models generalize, and thus how universal the identified important features are. Lastly, only one measure of feature importance - the ability of a feature to reduce impurity, was considered.

To improve confidence in our results, it may be useful to carry out regression using other techniques, for example linear, lasso or ridge regression. It may also be useful to use other measures of feature importance, for example, identifying important features using permutation based feature importance. Alternatively, frequent itemset mining could be performed, and features appearing in derived association rules could be compared with those features identified as important for regression. 

One means of broadening this study would be to augment the used data sets. For example, the study could be extended to include data from schools and states other than Massachusetts. The study may also be improved by using more fine-grain data to characterize school locations, as duplication resulted in reduced variability in features derived from American Survey Community Data. 


