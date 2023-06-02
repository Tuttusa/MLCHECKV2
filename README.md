![alt text](https://github.com/arnabsharma91/MLCHECKV2/blob/main/MLCHECK-logo.JPG)

# MLCHECK

This reporsitory contains the code for the tool MLCHECK which has been developed to test machine learning models with respect to specifiedproperties. 

## Usage
To use MLCHECK to test a property you need to first of all upload your model and give the path to the ```model_path``` parameter. MLCHECK needs the input output format of the given model to test, which could be given inside an XML file. There is a sample XML file ```dataInput.xml``` which shows you how to make such an XML file. If you have the training data format available in a dataframe, you could use the following code to generate an XML file in the specific format we require


