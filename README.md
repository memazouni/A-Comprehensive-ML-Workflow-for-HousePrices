# A-Comprehensive-ML-Workflow-for-HousePrices
 ## <div align="center">  A Comprehensive ML Workflow for House Prices</div>

There are plenty of **courses and tutorials** that can help you learn machine learning from scratch but here in **Kaggle**, I want to predict **House prices**  a popular machine learning Dataset as a comprehensive workflow with python packages. 
After reading, you can use this workflow to solve other real problems and use it as a template to deal with machine learning problems.  

<img src="http://s9.picofile.com/file/8338980150/House_price.png"></img>

---------------------------------------------------------------------
you can follow me on:
> ###### [ GitHub](https://github.com/mjbahmani)
> ###### [LinkedIn](https://www.linkedin.com/in/bahmani/)
> ###### [Kaggle](https://www.kaggle.com/mjbahmani/)
>###### you may  be interested have a look at it: [A Comprehensive Machine Learning Workflow with Python](http://https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)

 <a id="1"></a> <br>
## 1- Introduction
This is a **A Comprehensive ML Workflow for House Prices** data set, it is clear that everyone in this community is familiar with house prices dataset but if you need to review your information about the dataset please visit this [link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

I have tried to help **Fans of Machine Learning**  in Kaggle how to face machine learning problems. and I think it is a great opportunity for who want to learn machine learning workflow with python **completely**.

I have covered most of the methods that are implemented for house prices until **2018**, you can start to learn and review your knowledge about ML with a simple dataset and try to learn and memorize the workflow for your journey in Data science world.

I am open to getting your feedback for improving this **kernel**
<a id="2"></a> <br>
## 2- Machine Learning Workflow
If you have already read some [machine learning books](https://towardsdatascience.com/list-of-free-must-read-machine-learning-books-89576749d2ff). You have noticed that there are different ways to stream data into machine learning.

most of these books share the following steps:
*   Define Problem
*   Specify Inputs & Outputs
*   Exploratory data analysis
*   Data Collection
*   Data Preprocessing
*   Data Cleaning
*   Visualization
*   Model Design, Training, and Offline Evaluation
*   Model Deployment, Online Evaluation, and Monitoring
*   Model Maintenance, Diagnosis, and Retraining

**You can see my workflow in the below image** :
 <img src="http://s9.picofile.com/file/8338227634/workflow.png" />

<a id="3"></a> <br>
## 3- Problem Definition
I think one of the important things when you start a new machine learning project is Defining your problem.

Problem Definition has four steps that have illustrated in the picture below:
<img src="http://s8.picofile.com/file/8338227734/ProblemDefination.png">
<a id="4"></a> <br>
### 3-1 Problem Feature
we will use the house prices data set. This dataset contains information about house prices and the target value is:

* SalePrice

<img src="https://kaggle2.blob.core.windows.net/competitions/kaggle/5407/media/housesbanner.png"></img>
**Why am I  using House price dataset:**

1- This is a good project because it is so well understood.

2- Attributes are numeric and categurical so you have to figure out how to load and handle data.

3- It is a Regression problem, allowing you to practice with perhaps an easier type of supervised learning algorithm.

4- This is a perfect competition for data science students who have completed an online course in machine learning and are looking to expand their skill set before trying a featured competition. 

5-Creative feature engineering .


<a id="5"></a> <br>
### 3-2 Aim
It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 
<img src="https://totalbitcoin.org/wp-content/uploads/2015/12/Bitcoin-price-projections-for-2016.png"></img>
<a id="6"></a> <br>
### 3-3 Variables
The variables are :
* SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
* MSSubClass: The building class
* MSZoning: The general zoning classification
* LotFrontage: Linear feet of street connected to property
* LotArea: Lot size in square feet
* Street: Type of road access
* Alley: Type of alley access
* LotShape: General shape of property
* LandContour: Flatness of the property
* Utilities: Type of utilities available
* LotConfig: Lot configuration
* LandSlope: Slope of property
* Neighborhood: Physical locations within Ames city limits
* Condition1: Proximity to main road or railroad
* Condition2: Proximity to main road or railroad (if a second is present)
* BldgType: Type of dwelling
* HouseStyle: Style of dwelling
* OverallQual: Overall material and finish quality
* OverallCond: Overall condition rating
* YearBuilt: Original construction date
* YearRemodAdd: Remodel date
* RoofStyle: Type of roof
* RoofMatl: Roof material
* Exterior1st: Exterior covering on house
* Exterior2nd: Exterior covering on house (if more than one material)
* MasVnrType: Masonry veneer type
* MasVnrArea: Masonry veneer area in square feet
* ExterQual: Exterior material quality
* ExterCond: Present condition of the material on the exterior
* Foundation: Type of foundation
* BsmtQual: Height of the basement
* BsmtCond: General condition of the basement
* BsmtExposure: Walkout or garden level basement walls
* BsmtFinType1: Quality of basement finished area
* BsmtFinSF1: Type 1 finished square feet
* BsmtFinType2: Quality of second finished area (if present)
* BsmtFinSF2: Type 2 finished square feet
* BsmtUnfSF: Unfinished square feet of basement area
* TotalBsmtSF: Total square feet of basement area
* Heating: Type of heating
* HeatingQC: Heating quality and condition
* CentralAir: Central air conditioning
* Electrical: Electrical system
* 1stFlrSF: First Floor square feet
* 2ndFlrSF: Second floor square feet
*  LowQualFinSF: Low quality finished square feet (all floors)
* GrLivArea: Above grade (ground) living area square feet
* BsmtFullBath: Basement full bathrooms
* BsmtHalfBath: Basement half bathrooms
* FullBath: Full bathrooms above grade
* HalfBath: Half baths above grade
* Bedroom: Number of bedrooms above basement level
* Kitchen: Number of kitchens
* KitchenQual: Kitchen quality
* TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
* Functional: Home functionality rating
* Fireplaces: Number of fireplaces
* FireplaceQu: Fireplace quality
* GarageType: Garage location
* GarageYrBlt: Year garage was built
* GarageFinish: Interior finish of the garage
* GarageCars: Size of garage in car capacity
* GarageArea: Size of garage in square feet
* GarageQual: Garage quality
* GarageCond: Garage condition
* PavedDrive: Paved driveway
* WoodDeckSF: Wood deck area in square feet
* OpenPorchSF: Open porch area in square feet
* EnclosedPorch: Enclosed porch area in square feet
* 3SsnPorch: Three season porch area in square feet
* ScreenPorch: Screen porch area in square feet
* PoolArea: Pool area in square feet
* PoolQC: Pool quality
* Fence: Fence quality
* MiscFeature: Miscellaneous feature not covered in other categories
* MiscVal: $Value of miscellaneous feature
* MoSold: Month Sold
* YrSold: Year Sold
* SaleType: Type of sale
* SaleCondition: Condition of sale
<a id="7"></a> <br>
## 4- Inputs & Outputs
<img src="https://upload.wikimedia.org/wikipedia/commons/b/bc/Input-Output.JPG"></img>

<a id="8"></a> <br>
### 4-1 Inputs
* train.csv - the training set

<a id="9"></a> <br>

### 4-2 Outputs
* sale prices for every record in test.csv
