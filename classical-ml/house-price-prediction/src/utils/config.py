
##File Path
TRAIN_CSV_PATH = "data/raw/train.csv"
TEST_CSV_PATH = "data/raw/test.csv"

MODEL_OUTPUT_PATH = "models/final_model.pkl"
MODEL_PREDICTION_PATH = "reports/submission.csv"
EVAL_REPORT_PATH = "reports/evaluation_report.json"

#Model hyperparameters
SVR_PARAMS = {
    "C" : 0.5,
    "epsilon" : 0.01,
    "gamma" : 'auto',
    "kernel" : "rbf"
}


TARGET = 'SalePrice'
FILL_NA_COLS = ['FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','BsmtQual','BsmtCond','BsmtFinType1','BsmtExposure','BsmtFinType2']
FILL_ZERO_COLS = ['MasVnrArea','GarageYrBlt','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageArea','GarageCars']
FILL_NONE_COLS = ['MasVnrType']
FILL_MODE_COLS = ['Exterior1st','Exterior2nd','SaleType','Utilities','MSZoning','Functional','KitchenQual']
DROP_COLS = ['Id','Fence','Alley','MiscFeature','PoolQC']
