# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This is an Income Classification model using Random Forest Classifier (from sklearn open source Python library)

## Intended Use

This model is designed to predict whether an individual's income exceeds $50K/year based on demographic and employment-related attributes.

## Training Data

The model was trained on the provided census data in the csv.  The raw data had 32,562 records before cleaning.
The attributes of the data include:

- age
- workclass
- fnlgt (final weight)
- education
- education-num
- marital-status
- occupation
- relationship
- race
- sex
- capital-gain
- capital-loss
- hours-per-week
- native-country
- salary

The target variable is income, categorized as either '>50K' or '<=50K'. The dataset was preprocessed to handle missing values, encode categorical variables, and normalize numerical features.

## Evaluation Data

The model was evaluated using a test set comprising 6512 (20% of dataset) instances, separated from the original dataset prior to training. This test set maintains the same feature structure and preprocessing steps as the training data.

## Metrics

_Please include the metrics used and your model's performance on those metrics._
The model's performance was assessed using the following metrics:​

- Precision: Proportion of positive identifications that were actually correct.​
- Recall: Proportion of actual positives that were correctly identified.​
- F1 Score: Harmonic mean of precision and recall.​

On the test set, the model achieved:​

- Precision: 0.7419​
- Recall: 0.6384
- F1 Score: 0.6863

## Ethical Considerations

While the model provides insights based on demographic and employment data, it is crucial to consider the following ethical aspects:​

Bias and Fairness: The model may inadvertently reflect biases present in the training data, particularly concerning sensitive attributes like race, sex, and marital status. Users should analyze and mitigate any unfair biases before deployment.​

Privacy: The dataset contains personal information. Ensure that any application of the model complies with data protection regulations and respects individuals' privacy.​

Interpretability: Given the complexity of Random Forest models, interpreting individual predictions can be challenging. Users should employ appropriate techniques to explain model decisions when necessary.

## Caveats and Recommendations

- Generalization: The model's performance is contingent on the representativeness of the training data. Caution should be exercised when applying it to populations or contexts that differ significantly from the training dataset.​  The dataset is of US origin hence the caution.  Applying similar processing to data from other countries might produce simlar results.  This is not guaranteed.

- Temporal Validity: The socioeconomic factors influencing income can change over time. Regularly update and validate the model to maintain its relevance and accuracy.​  Innovation and policy changes and implementations in the source country i.e. US can affect subsequent models based on new data.

- Complementary Analysis: Use the model as a supplementary tool alongside other analyses and expert judgment. Avoid relying solely on this model predictions for critical decisions.​

- Continuous Monitoring: Implement monitoring mechanisms to detect and address any degradation in model performance or emerging biases during its operational use.​

By adhering to these considerations and recommendations, users can responsibly leverage the model's capabilities while minimizing potential risks.
