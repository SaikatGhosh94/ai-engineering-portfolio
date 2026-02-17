import joblib
import pandas as pd
import numpy as np
from src.utils.config import MODEL_OUTPUT_PATH,TEST_CSV_PATH,MODEL_PREDICTION_PATH

def predict(test_path,model_path,output_path):

    pipeline = joblib.load(model_path)

    test_df = pd.read_csv(test_path)

    log_preds = pipeline.predict(test_df)

    preds = np.expm1(log_preds)

    submission = pd.DataFrame({"Id":test_df["Id"],
                               "SalePrice":preds})
    
    submission.to_csv(output_path,index=False)

if __name__ == "__main__":
    predict(TEST_CSV_PATH,MODEL_OUTPUT_PATH,MODEL_PREDICTION_PATH)