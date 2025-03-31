# MTEA
# Where and How: Mining Convertible Outlying Aspect for Outlier Interpretation

Since GAOI/explainers/AETabularMM.py and GAOI/explainers/MaskingModelExplainer.py are the core code of the META model, if this paper is lucky enough to be accepted, we will publicly release it upon acceptance.

# Structure

datasets: real-world datasets in csv format, the last column is label indicating each line is an outlier or inlier; ground_truth outlier interpretation annotations of real-world datasets.

explainers: the generative component 

models: the adversarial component

utils: configuration and default hype-parameters

test_runner_independent.py: generate synthetic datasets

test_runner_real.py: main script to run the experiments

# How to use?

For META

1.modify variant dataset in argument_parser.py 

2.According to dataset in argument_parser.py, modify g_truth_df in test_runner_real.py

3.use python test_runner_real.py

4.the results can be found in test/logs folder


# Requirements

tensorflow 2.6.2

numpy 1.19.5

python 3.7.12

scikit-learn 1.0.2

pandas 1.3.5

