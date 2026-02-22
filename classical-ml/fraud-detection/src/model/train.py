from src.data.load_data import load
from src.features.build_features import build
from sklearn.linear_model import LogisticRegression
from src.utils.config import MODEL_OUTPUT_PATH,MODEL_REPORT_PATH
import joblib
import pandas as pd
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,RocCurveDisplay
import matplotlib.pyplot as plt
def train():
    X,y = load()
    X_train,y_train,X_test,y_test=build(X,y)
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train,y_train)

    joblib.dump(model,MODEL_OUTPUT_PATH)

    print("Model trained and saved successfully")
    return model,X_test,y_test


def predict(model,X_test):
    return model.predict(X_test)

def evaluate(y_test,pred):

    print(classification_report(y_test,pred))

    classfication_report_path = MODEL_REPORT_PATH / "logistic_regression_classification_report.txt"
    with open(classfication_report_path,"w") as f:
        f.write(f"--- Logistic Regression Classification Report ---\n\n")
        f.write(classification_report(y_test,pred))

    confusion_matrix_path = MODEL_REPORT_PATH / "logistic_regression_confusion_matrix.png"
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test,pred)
    plt.title("Logistic Regresion Confusion matrix")
    plt.savefig(confusion_matrix_path, bbox_inches='tight', dpi=300) 
    plt.close()

    print("\n"*2)

    roc_path = MODEL_REPORT_PATH / "logistic_regression_roc.png"
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test,pred)
    plt.title("Logistic Regression ROC")
    plt.savefig(roc_path, bbox_inches='tight', dpi=300) 
    plt.close()

    print("Evaluation reports stored successfully")

if __name__ == '__main__':
    model,X_test,y_test = train()
    pred = predict(model,X_test)
    evaluate(y_test,pred)


