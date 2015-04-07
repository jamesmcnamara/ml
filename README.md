## Learn.py
_By James McNamara_

Learn.py is a general purpose ETL and machine learning library written in python3 with a focus on lazy, functional style. It currently includes various decision trees, regression tools, and text classifiers and work has already begun on neural nets, support vector machines, and EM clustering.

The required libraries are included in `requirements.txt`, and can be installed with:   

```
	pip install -r requirements.txt
```  
#### Examples
Most classes support the same API, and thus can be used through:

```
from ml.module import MLClass

clf = MLClass(data=my_training_data, results=Training_results)
predictions = clf.predict(test_data)
```
It should be noted that output is an iterable, and is thus single use, and calculated by need.

### CLI

The project has a command line interface accessible through learn.py:  

```
python learn.py [-h] [-r RANGE RANGE RANGE] [-m META] 
			    [-cv CROSS] [-t TREE] [-d] [-cf] [-b] 
			    infile
```
#### Positional Arguments
|Name      |					Usage			|
|----------|----------------------------|
|_infile_| CSV file with training data|

#### Optional Arguments
|Name      |					Usage			|
|----------|----------------------------|
| _-h, --help_| Show this help message and exit
| _-r RANGE RANGE RANGE_, <br>_--range RANGE RANGE RANGE_|Range of &#951; values to use for cross validation. The first value is start, the second is end, and the last is interval|
|_-m META,<br> --meta META_ | Meta file containing JSON formatted descriptions of the data
|_-cv CROSS, <br>--cross CROSS_ | Set the parameter for k-fold cross validation. Default 10.|
|_-t TREE, <br> --tree TREE_ | What type of decision tree to build for the data. Options are 'entropy', 'regression', or 'categorical'. Default 'entropy'
|_-d, --debug_ | Use sci-kit learn instead of learn.py, to test that the behavior is correct|
|_-cf, --with-confusion_| Include a confusion matrix in the output
|_-b, --binary-splits_ |Convert a multi-way categorical matrix to a binary matrix


#### Examples

Perform 10-fold cross validation on the iris dataset over &#951; mins of 5, 10, 15, 20 & 25:  

```
    python learn.py -r 5 25 5 data/iris.csv
```

Generate confusion matricies for &#951; mins of 5 10 15 over the mushroom dataset using multiway splits:  

```  
    python learn.py -r 5 15 5 -t categorical -cf data/mushroom.csv
```  

Convert the mushroom dataset to a binary dataset and perform cross validation at 1-10:   

```
    python learn.py -r 1 10 1 -t categorical -b data/mushroom.csv
```  

Regress the housing dataset using 15-fold cross validation over &#951; of 5, 10 & 15:  

```
    python learn.py -r 5 15 5 -t regression -cv 15 data/housing.csv
```