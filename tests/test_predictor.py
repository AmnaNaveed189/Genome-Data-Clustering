import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.predictor import predict_new_patient, save_all_models, train_subtype_predictor


def test_predictor_training_save_and_predict(tmp_path):
    X, y = make_blobs(n_samples=300, centers=3, n_features=10, random_state=42)
    config = {}

    clf = train_subtype_predictor(X, y, config)
    assert clf is not None

    scaler = StandardScaler().fit(np.log1p(np.abs(X)))
    pca = PCA(n_components=10, random_state=42).fit(scaler.transform(np.log1p(np.abs(X))))
    save_all_models(scaler, pca, clf, clf, save_dir=str(tmp_path))

    sample = np.abs(X[0])
    subtype, confidence = predict_new_patient(sample, models_dir=str(tmp_path))
    assert isinstance(subtype, int)
    assert 0.0 <= confidence <= 1.0
