from catboost import CatBoostClassifier
import pandas as pd


def main():
    labels = ['VETER', 'SHKVAL', 'METEL', 'DOZD', 'SNEG', 'GRAD', 'TUMAN', 'GOLOLED']
    model = CatBoostClassifier()
    model.load_model("../models_weights/best_catboost.cbm")

    data = pd.read_csv("../data/X_test_good_features.csv")
    y_pred = model.predict(data)
    pd.DataFrame(y_pred, columns=labels).to_csv("output.csv", index=False)
    

if __name__ == "__main__":
    main()
