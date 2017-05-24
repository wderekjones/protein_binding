from utils import combine_positive_negative_data
from utils import load_data_h5

X_p, y_p = load_data_h5("data/ml_pro_features_labels.h5",mode=1)
X_n, y_n = load_data_h5("data/ml_pro_features_labels.h5",mode=0)

X = combine_positive_negative_data(X_p,X_n)
y = combine_positive_negative_data(y_p,y_n)

