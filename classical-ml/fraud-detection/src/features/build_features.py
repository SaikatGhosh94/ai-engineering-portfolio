from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from src.utils.config import RANDOM_STATE
def build(X,y):

    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.33, random_state=RANDOM_STATE)
    smote = SMOTE(sampling_strategy=0.2,random_state = RANDOM_STATE)

    X_train_res,y_train_res = smote.fit_resample(X_train,y_train)
    return X_train_res,y_train_res,X_test,y_test


