"""
Simplified cross-validation module for linear probing experiments.
Performs cross-validation on embeddings with logistic regression.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve
from scipy.stats import loguniform
from scipy.special import expit
import joblib
import copy
try:
    from .utils import LabelColumn
except ImportError:
    from utils import LabelColumn


class MetricsCalculator:
    """Handles calculation of various performance metrics."""
    
    @staticmethod
    def confusion_matrix_from_scores(y_true, y_scores):
        y_pred = (y_scores[:, 1] > 0.5).astype(int)
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def specificity_at_fixed_sensitivity(y_true, y_scores, sensitivity_threshold=0.999):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores[:, 1])
        sensitivity_index = next(i for i, sens in reversed(list(enumerate(tpr))) 
                               if sens <= sensitivity_threshold)
        return 1 - fpr[sensitivity_index]

    @staticmethod
    def precision_at_fixed_sensitivity(y_true, y_scores, fixed_sensitivity=0.99):
        precisions, sensitivities, thresholds = precision_recall_curve(y_true, y_scores[:, 1])
        idx = next(i for i, sens in enumerate(sensitivities) if sens >= fixed_sensitivity)
        return precisions[idx]

    @staticmethod
    def calculate_metric(y_true, y_scores, metric_name):
        """Calculate a specific metric given true labels and prediction scores."""
        cm = MetricsCalculator.confusion_matrix_from_scores(y_true, y_scores)
        
        metrics_map = {
            'specificity': lambda: cm[0, 0] / (cm[0, 0] + cm[0, 1]),
            'sensitivity': lambda: cm[1, 1] / (cm[1, 1] + cm[1, 0]),
            'valeur_predictive_positive': lambda: cm[1, 1] / (cm[1, 1] + cm[0, 1]),
            'valeur_predictive_negative': lambda: cm[0, 0] / (cm[0, 0] + cm[1, 0]),
        }
        
        if metric_name in metrics_map:
            return metrics_map[metric_name]()
        
        return None


def load_embeddings_and_labels(dataset_path, label_csv, label_name='label', patient_col=None):
    """Load embeddings and labels, returning matched data."""
    embs = np.load(dataset_path / 'embeddings.npy')
    ids = np.load(dataset_path / 'ids.npy')
    dico_embs = {id_: embs[i] for i, id_ in enumerate(ids)}
    
    df_csv = pd.read_csv(label_csv)
    labels = df_csv.set_index('ID').to_dict()[label_name]
    
    # Match IDs between embeddings and labels
    matched_ids = list(set(dico_embs.keys()) & set(labels.keys()))
    X = np.array([dico_embs[id_] for id_ in matched_ids])
    y = np.array([labels[id_] for id_ in matched_ids])
    
    # Handle patient information if provided
    patients = None
    if patient_col and patient_col in df_csv.columns:
        patient_dict = df_csv.set_index('ID').to_dict()[patient_col]
        patients = np.array([patient_dict[id_] for id_ in matched_ids])
    
    return X, y, matched_ids, patients


def get_cv_splits(y, cv_type, n_splits=5, patients=None, test_splits=None):
    """Generate cross-validation splits based on type."""
    if cv_type == 'fixed':
        if test_splits is None:
            raise ValueError("test_splits required for fixed CV")
        for test_val in np.unique(test_splits):
            train_idx = np.where(test_splits != test_val)[0]
            test_idx = np.where(test_splits == test_val)[0]
            yield train_idx, test_idx
    elif cv_type == 'random':
        if patients is not None:
            skf = StratifiedGroupKFold(n_splits=n_splits)
            yield from skf.split(np.zeros(len(y)), y, groups=patients)
        else:
            skf = StratifiedKFold(n_splits=n_splits)
            yield from skf.split(np.zeros(len(y)), y)
    else:
        raise ValueError(f'Unknown cv_type: {cv_type}')


def compute_scores(clf, X, y, score_types):
    """Compute multiple scores for a fitted classifier."""
    if isinstance(score_types, str):
        score_types = [score_types]
    
    scores = {}
    y_proba = clf.predict_proba(X)
    y_pred = clf.predict(X)
    
    for score_type in score_types:
        if score_type == 'roc_auc_score':
            if len(np.unique(y)) == 2:
                scores[score_type] = metrics.roc_auc_score(y, y_proba[:, 1])
            else:
                scores[score_type] = metrics.roc_auc_score(y, y_proba, multi_class='ovr')
        elif score_type == 'f1_score':
            scores[score_type] = metrics.f1_score(y, y_pred, average='macro')
        elif score_type in ['specificity', 'sensitivity', 'valeur_predictive_positive', 'valeur_predictive_negative']:
            scores[score_type] = MetricsCalculator.calculate_metric(y, y_proba, score_type)
        elif score_type == 'precision_at_fixed_sensitivity':
            scores[score_type] = MetricsCalculator.precision_at_fixed_sensitivity(y, y_proba)
        elif score_type == 'specificity_at_fixed_sensitivity':
            scores[score_type] = MetricsCalculator.specificity_at_fixed_sensitivity(y, y_proba)
        else:
            # Try to get metric from sklearn.metrics
            try:
                scores[score_type] = getattr(metrics, score_type)(y, y_pred)
            except AttributeError:
                print(f"Warning: Unknown score type {score_type}")
    
    return scores


def cross_validate(dataset_path, label_csv, cv_type='fixed', label_name='label', 
                  n_splits=5, score_types='roc_auc_score', patient_col=None,
                  C_fixed=15, use_nested_cv=False, n_iter=10):
    """
    Perform cross-validation on embeddings.
    
    Args:
        dataset_path: Path to dataset directory containing embeddings.npy and ids.npy
        label_csv: Path to CSV file with labels
        cv_type: 'fixed' or 'random'
        label_name: Column name for labels
        n_splits: Number of folds for random CV
        score_types: Score type(s) to calculate
        patient_col: Column name for patient info (for stratified group CV)
        C_fixed: Fixed regularization parameter (if not using nested CV)
        use_nested_cv: Whether to use nested CV for hyperparameter tuning
        n_iter: Number of iterations for random search in nested CV
    
    Returns:
        DataFrame with cross-validation results and list of trained classifiers
    """
    # Load data
    X, y, matched_ids, patients = load_embeddings_and_labels(
        Path(dataset_path), label_csv, label_name, patient_col
    )
    
    print(f"Loaded {len(matched_ids)} samples with {X.shape[1]} features")
    
    # Normalize features
    normalizer = Normalizer()
    X = normalizer.fit_transform(X)
    
    # Get test splits if using fixed split based on label type
    test_splits = None
    uses_fixed_split = LabelColumn.uses_fixed_split(label_name)

    if cv_type == 'fixed' or uses_fixed_split:
        # For 'label' column, always use fixed split; for others, respect cv_type parameter
        if uses_fixed_split or cv_type == 'fixed':
            df_csv = pd.read_csv(label_csv)
            if 'test' not in df_csv.columns:
                if uses_fixed_split:
                    raise ValueError(f"'test' column required for label '{label_name}' which uses fixed split")
                else:
                    raise ValueError("'test' column required for fixed CV")
            test_dict = df_csv.set_index('ID').to_dict()['test']
            test_splits = np.array([test_dict[id_] for id_ in matched_ids])
            # Override cv_type to ensure we use fixed splitting
            cv_type = 'fixed'
    
    # Perform cross-validation
    scores = []
    classifiers = []
    predictions = []
    
    splits = get_cv_splits(y, cv_type, n_splits, patients, test_splits)
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        if use_nested_cv:
            # Nested CV with hyperparameter tuning
            param_dist = {'C': loguniform(1e-3, 1e3)}
            random_search = RandomizedSearchCV(
                LogisticRegression(max_iter=10000, class_weight='balanced'),
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring='roc_auc',
                cv=n_splits
            )
            random_search.fit(X_train, y_train)
            clf = random_search.best_estimator_
        else:
            # Fixed regularization
            clf = LogisticRegression(C=C_fixed, max_iter=10000, class_weight='balanced')
            clf.fit(X_train, y_train)
        
        # Compute scores
        fold_scores = compute_scores(clf, X_test, y_test, score_types)
        fold_scores['fold'] = fold
        scores.append(fold_scores)
        
        # Store classifier
        classifiers.append(clf)
        
        # Store predictions
        y_proba = clf.predict_proba(X_test)[:, 1]
        test_ids = [matched_ids[i] for i in test_idx]
        fold_predictions = [
            {'ID': id_, 'label': label, 'proba': prob, 'fold': fold}
            for id_, label, prob in zip(test_ids, y_test, y_proba)
        ]
        predictions.extend(fold_predictions)
    
    # Create results DataFrame
    scores_df = pd.DataFrame(scores)
    
    # Add summary statistics
    numeric_cols = [col for col in scores_df.columns if col != 'fold']
    mean_row = scores_df[numeric_cols].mean().to_dict()
    mean_row['fold'] = 'mean'
    std_row = scores_df[numeric_cols].std().to_dict()
    std_row['fold'] = 'std'
    
    summary_df = pd.concat([
        scores_df,
        pd.DataFrame([mean_row, std_row])
    ], ignore_index=True)
    
    predictions_df = pd.DataFrame(predictions)
    
    return summary_df, classifiers, predictions_df, normalizer


def test_classifier(dataset_path, label_csv, classifier_path, score_types='roc_auc_score', 
                   label_name='label'):
    """
    Test a saved classifier on a dataset.
    
    Args:
        dataset_path: Path to test dataset
        label_csv: Path to CSV with test labels
        classifier_path: Path to saved classifier
        score_types: Score types to calculate
        label_name: Column name for labels
    
    Returns:
        DataFrame with test results and predictions
    """
    # Load data
    X, y, matched_ids, _ = load_embeddings_and_labels(
        Path(dataset_path), label_csv, label_name
    )
    
    # Load saved model data
    model_data = joblib.load(classifier_path)
    
    # Handle different saved formats
    if isinstance(model_data, dict):
        # New format: {'classifiers': [...], 'normalizer': ..., 'label_name': ...}
        classifiers = model_data.get('classifiers', [])
        saved_normalizer = model_data.get('normalizer')
        if not isinstance(classifiers, list):
            classifiers = [classifiers]
    else:
        # Legacy format: just classifiers
        classifiers = model_data
        if not isinstance(classifiers, list):
            classifiers = [classifiers]
        saved_normalizer = None
    
    # Apply normalization
    if saved_normalizer is not None:
        # Use the saved normalizer
        X = saved_normalizer.transform(X)
    else:
        # Fallback to fitting a new normalizer (for backward compatibility)
        normalizer = Normalizer()
        X = normalizer.fit_transform(X)
    
    # Ensemble prediction if multiple classifiers
    y_proba_all = []
    for clf in classifiers:
        y_proba_all.append(clf.predict_proba(X))
    
    y_proba = np.mean(y_proba_all, axis=0)
    
    # Compute scores using ensemble
    class EnsembleClassifier:
        def __init__(self, y_proba, classes):
            self.y_proba_ = y_proba
            self.classes_ = classes
        
        def predict_proba(self, X):
            return self.y_proba_
        
        def predict(self, X):
            return self.classes_[np.argmax(self.y_proba_, axis=1)]
    
    ensemble_clf = EnsembleClassifier(y_proba, classifiers[0].classes_)
    scores = compute_scores(ensemble_clf, X, y, score_types)
    
    # Create predictions DataFrame
    predictions = []
    for i, (id_, label) in enumerate(zip(matched_ids, y)):
        pred_dict = {'ID': id_, 'label': label}
        for class_idx in range(y_proba.shape[1]):
            pred_dict[f'proba_{class_idx}'] = y_proba[i, class_idx]
        predictions.append(pred_dict)
    
    predictions_df = pd.DataFrame(predictions)
    scores_df = pd.DataFrame([scores])
    
    return scores_df, predictions_df


def save_results(scores_df, classifiers, predictions_df, normalizer, save_dir, 
                name_prefix, label_name):
    """Save cross-validation results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = f"{name_prefix}_{label_name}"
    
    # Save scores
    scores_df.to_csv(save_dir / f"cv_scores_{base_name}.csv", index=False)
    
    # Save predictions
    predictions_df.to_csv(save_dir / f"cv_predictions_{base_name}.csv", index=False)
    
    # Save classifiers and normalizer
    model_data = {
        'classifiers': classifiers,
        'normalizer': normalizer,
        'label_name': label_name
    }
    joblib.dump(model_data, save_dir / f"model_{base_name}.pkl")
    
    print(f"Results saved to {save_dir}")
    return save_dir / f"model_{base_name}.pkl"