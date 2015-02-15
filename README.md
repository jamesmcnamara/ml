## Learn.py
_By James McNamara_

Learn.py is a general purpose ETL and machine learning library written in python3. At the moment, it includes only various decision trees and regression tools, but development is continuing every day. Work has already begun on random forests, neural nets and support vector machines.

The required libraries are included in `requirements.txt`, and can be installed with:   

```
	pip install -r requirements.txt
```  

The project has a command line interface accessible through cli.py:  

```
python cli.py [-h] [-r RANGE RANGE RANGE] [-m META] 
			  [-cv CROSS] [-t TREE] [-d] [-cf] [-b] 
			  infile
```
### Positional Arguments
|Name      |					Usage			|
|----------|----------------------------|
|_infile_| CSV file with training data|

### Optional Arguments
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


### Examples

1. Perform 10-fold cross validation on the iris dataset over &#951; mins of 5, 10, 15, 20 & 25:  
```
    python cli.py -r 5 25 5 data/iris.csv
```

2. Generate confusion matricies for &#951; mins of 5 10 15 over the mushroom dataset using multiway splits:  
```
    python cli.py -r 5 15 5 -t categorical -cf data/mushroom.csv
```
3. Convert the mushroom dataset to a binary dataset and perform cross validation at 1-10: 
```
    python cli.py -r 1 10 1 -t categorical -b data/mushroom.csv
```
4. Regress the housing dataset using 15-fold cross validation over &#951; of 5, 10 & 15:  
```
    python cli.py -r 5 15 5 -t regression -cv 15 data/housing.csv
```