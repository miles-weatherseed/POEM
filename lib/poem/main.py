import argparse
import numpy as np
import pandas as pd
import os
import yaml
import random
from typing import Dict, Any
import tensorflow as tf
import joblib
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD, Adam
from poem.utils.utils import make_poem_input_layer, VALID_ENCODINGS
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    precision_recall_curve,
    auc,
)
from lightgbm import LGBMClassifier
import lightgbm as lgb

np.set_printoptions(precision=4)

current_dir = os.path.dirname(os.path.abspath(__file__))


# function to convert command line inputs into lower case
def case_insensitive_mode(value):
    return value.casefold()


def main():
    parser = argparse.ArgumentParser(
        description="Provide data to train or test POEM."
    )

    # Adding -p for input_peptides (CSV file)
    parser.add_argument(
        "-d",
        "--input_data",
        help="Path to the input data CSV file.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-m",
        "--mode",
        help="Whether to train a new model or test one.",
        required=True,
        type=case_insensitive_mode,  # Convert to lowercase (case-insensitive)
        choices=["train", "test"],  # Restricted choices
    )

    parser.add_argument(
        "-c",
        "--config_path",
        help="Path to config file.",
        type=str,
        required=False,
        default=os.path.join(current_dir, "poem_settings.yml"),
    )
    # Parse the arguments
    args = parser.parse_args()
    config = yaml.safe_load(
        open(
            args.config_path,
            "r",
        )
    )
    # check config will work
    config = validate_yaml_string_inputs(config)
    config = validate_yaml_numerical_inputs(config)

    fix_random_states(config["training"]["random_seed"])

    # load in training or test data
    data = pd.read_csv(args.input_data)
    # data = data.loc[data.random == 0]
    # data.reset_index(inplace=True)

    # if this is a valid yaml then we can move on to train/test
    if args.mode == "train":
        train_poem_LGBM(data, config)
    elif args.mode == "test":
        data = test_poem(data, config)
        data.to_csv(f"{args.input_data[:-4]}_poem_preds.csv", index=False)

    return


def train_poem_LGBM(training_data: pd.DataFrame, config: Dict[str, Any]):

    # Create input data
    print(
        "Fold",
        "AUROC",
        "logloss",
        "avg_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "mcc",
        "kappa",
        "pr_auc",
    )
    poem_input = make_poem_input_layer(
        training_data["mhc_i"].values,
        training_data["length"].values,
        training_data["peptide"].values,
        training_data["predicted_pmhc"].values,
        config["dataset"]["length_encoding"],
        config["dataset"]["mhci_encoding"],
        config["dataset"]["mhci_sequence"],
        config["dataset"]["tcr_encoding"],
    )
    poem_target = training_data["immunogenicity"].values

    # Data normalization
    if config["dataset"]["normalization"] == "minmax":
        scaler = MinMaxScaler()
    elif config["dataset"]["normalization"] == "standard":
        scaler = StandardScaler()
    else:  # no scaling
        scaler = None

    # Create K folds
    if config["training"]["stratified"]:
        kf = StratifiedKFold(
            n_splits=config["training"]["kfolds"],
            shuffle=True,
            random_state=config["training"]["random_seed"],
        )
    else:
        kf = KFold(
            n_splits=config["training"]["kfolds"],
            shuffle=True,
            random_state=config["training"]["random_seed"],
        )

    # tuned_hyperparams = {
    #     "n_estimators": 1000,
    #     "learning_rate": 0.04504908594871347,
    #     "num_leaves": 2759,
    #     "max_depth": 8,
    #     "min_data_in_leaf": 1280,
    #     "max_bin": 291,
    #     "min_gain_to_split": 0.4991435685939899,
    #     "bagging_fraction": 0.8,
    #     "bagging_freq": 1,
    #     "feature_fraction": 0.6000000000000001,
    # }
    tuned_hyperparams = {
        "n_estimators": 1000,
        "learning_rate": 0.0708014875222241,
        "num_leaves": 383,
        "max_depth": 2,
        "min_data_in_leaf": 4100,
        "max_bin": 232,
        "min_gain_to_split": 4.145318017624779,
        "bagging_fraction": 0.6000000000000001,
        "bagging_freq": 4,
        "feature_fraction": 1.0,
    }
    # Test K-Fold performance
    for i, (train_index, test_index) in enumerate(
        kf.split(poem_input, poem_target)
    ):
        X_train = poem_input[train_index]
        y_train = poem_target[train_index]

        X_train_t, X_train_v, y_train_t, y_train_v = train_test_split(
            X_train,
            y_train,
            test_size=config["training"]["validation_split"],
            shuffle=True,
            random_state=config["training"]["random_seed"],
        )

        # Create LightGBM classifier
        model = LGBMClassifier(
            objective="binary",
            # class_weight=class_weights,
            random_state=config["training"]["random_seed"],
            verbosity=-1,
            **tuned_hyperparams,
        )
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        # Early stopping
        model.fit(
            X_train_t,
            y_train_t,
            eval_set=[(X_train_v, y_train_v)],
            eval_metric="binary_logloss",
            callbacks=callbacks,
        )

        X_test = poem_input[test_index]
        true_labels = poem_target[test_index]
        test_scores = model.predict_proba(X_test)[:, 1]
        thresholds = np.linspace(0.05, 0.5, 46)
        best_threshold = 0.5
        best_metric = -1

        for threshold in thresholds:
            pred_labels = (test_scores >= threshold).astype(int)
            metric_value = f1_score(
                true_labels, pred_labels
            )  # Change this to the metric you want

            if metric_value > best_metric:
                best_metric = metric_value
                best_threshold = threshold
        predicted_labels = (test_scores >= best_threshold).astype(int)

        # Calculate performance metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(
            true_labels, predicted_labels, zero_division=0
        )
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        balanced_accuracy = balanced_accuracy_score(
            true_labels, predicted_labels
        )
        mcc = matthews_corrcoef(true_labels, predicted_labels)
        kappa = cohen_kappa_score(true_labels, predicted_labels)
        logloss = log_loss(true_labels, test_scores)
        avg_precision = average_precision_score(true_labels, test_scores)
        roc_auc = roc_auc_score(true_labels, test_scores)
        precision_p, recall_p, _ = precision_recall_curve(
            true_labels, test_scores
        )

        print(
            i,
            roc_auc,
            logloss,
            avg_precision,
            accuracy,
            precision,
            recall,
            f1,
            balanced_accuracy,
            mcc,
            kappa,
            auc(recall_p, precision_p),
        )

    # Finally, train and save the model on all data
    X_train = poem_input
    y_train = poem_target
    X_train_t, X_train_v, y_train_t, y_train_v = train_test_split(
        X_train,
        y_train,
        test_size=config["training"]["validation_split"],
        # shuffle=True,
        random_state=config["training"]["random_seed"],
    )

    # Create LightGBM classifier
    model = LGBMClassifier(
        objective="binary",
        random_state=config["training"]["random_seed"],
        verbosity=-1,
        **tuned_hyperparams,
    )
    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
    # Early stopping
    model.fit(
        X_train_t,
        y_train_t,
        eval_set=[(X_train_v, y_train_v)],
        eval_metric="binary_logloss",
        callbacks=callbacks,
    )
    test_scores = model.predict_proba(X_train)[:, 1]
    thresholds = np.linspace(0.05, 0.5, 46)
    best_threshold = 0.5
    best_metric = -1

    for threshold in thresholds:
        pred_labels = (test_scores >= threshold).astype(int)
        metric_value = f1_score(
            y_train, pred_labels
        )  # Change this to the metric you want

        if metric_value > best_metric:
            best_metric = metric_value
            best_threshold = threshold
    print(best_threshold)

    # save the trained model
    joblib.dump(model, "trained_models/LightGBM_POEM.pkl")
    return


def train_poem(training_data: pd.DataFrame, config: Dict[str, Any]):
    # We re-train POEM using mechanistic output and user-defined settings
    # Create input data
    print(
        "Fold",
        "AUROC",
        "logloss",
        "avg_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "mcc",
        "kappa",
    )
    poem_input = make_poem_input_layer(
        training_data["mhc_i"].values,
        training_data["length"].values,
        training_data["peptide"].values,
        training_data["predicted_pmhc"].values,
        config["dataset"]["length_encoding"],
        config["dataset"]["mhci_encoding"],
        config["dataset"]["mhci_sequence"],
        config["dataset"]["tcr_encoding"],
    )

    poem_target = training_data["immunogenicity"].values

    if config["dataset"]["normalization"] == "minmax":
        scaler = MinMaxScaler()
    elif config["dataset"]["normalization"] == "standard":
        scaler = StandardScaler()
    else:  # no scaling
        scaler = None

    # create K folds
    if config["training"]["stratified"]:
        kf = StratifiedKFold(
            n_splits=config["training"]["kfolds"],
            shuffle=True,
            random_state=config["training"]["random_seed"],
        ).split(poem_input, poem_target)
    else:
        kf = KFold(
            n_splits=config["training"]["kfolds"],
            shuffle=True,
            random_state=config["training"]["random_seed"],
        ).split(poem_input, poem_target)

    # test K-Fold performance
    for i, (train_index, test_index) in enumerate(kf):
        X_train = poem_input[train_index]
        y_train = poem_target[train_index]

        # neg, pos = np.bincount(y_train)
        # total = neg + pos

        # weight_for_0 = (1 / np.sum(y_train == 0)) * (len(y_train) / 2.0)
        # weight_for_1 = (1 / np.sum(y_train == 1)) * (len(y_train) / 2.0)

        # class_weights = {0: weight_for_0, 1: weight_for_1}

        if config["training"]["early_stopping"]["enabled"]:
            X_train_t, X_train_v, y_train_t, y_train_v = train_test_split(
                X_train,
                y_train,
                test_size=config["training"]["validation_split"],
                shuffle=True,
                random_state=config["training"]["random_seed"],
            )
            if scaler is not None:
                X_train_t = scaler.fit_transform(X_train_t)
                X_train_v = scaler.transform(X_train_v)

            neg, pos = np.bincount(y_train_t)
            total = neg + pos
            weight_for_0 = (1 / neg) * (total / 2.0)
            weight_for_1 = (1 / pos) * (total / 2.0)

            class_weights = {0: weight_for_0, 1: weight_for_1}

            initial_bias = np.log([pos / neg])
            model = create_mlp(
                config, X_train.shape[1], output_bias=initial_bias
            )
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=config["training"]["early_stopping"]["patience"],
                restore_best_weights=config["training"]["early_stopping"][
                    "restore_best_weights"
                ],
            )
            model.fit(
                X_train_t,
                y_train_t,
                epochs=config["training"]["epochs"],
                callbacks=[early_stopping],
                verbose=0,
                validation_data=(X_train_v, y_train_v),
                class_weight=class_weights,
            )
        else:
            model = create_mlp(
                config, X_train.shape[1], output_bias == np.log([0.01])
            )
            if scaler is not None:
                X_train = scaler.fit_transform(X_train)
            model.fit(
                X_train,
                y_train,
                epochs=config["training"]["epochs"],
                verbose=0,
            )
        if scaler is not None:
            X_test = scaler.fit_transform(poem_input[test_index])
        else:
            X_test = poem_input[test_index]
        test_scores = model.predict(X_test, verbose=0)
        threshold = (
            0.5  # You can adjust the threshold based on your requirements
        )
        predicted_labels = (test_scores >= threshold).astype(int)
        true_labels = poem_target[test_index]

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(
            true_labels, predicted_labels, zero_division=0
        )
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        balanced_accuracy = balanced_accuracy_score(
            true_labels, predicted_labels
        )
        mcc = matthews_corrcoef(true_labels, predicted_labels)
        kappa = cohen_kappa_score(true_labels, predicted_labels)
        logloss = log_loss(true_labels, test_scores)
        avg_precision = average_precision_score(true_labels, test_scores)

        print(
            i,
            roc_auc_score(poem_target[test_index], test_scores),
            logloss,
            avg_precision,
            accuracy,
            precision,
            recall,
            f1,
            balanced_accuracy,
            mcc,
            kappa,
        )

    # finally train and save model on all data
    X_train = poem_input
    y_train = poem_target
    if config["training"]["early_stopping"]["enabled"]:
        X_train_t, X_train_v, y_train_t, y_train_v = train_test_split(
            X_train,
            y_train,
            test_size=config["training"]["validation_split"],
            shuffle=True,
            random_state=config["training"]["random_seed"],
        )
        if scaler is not None:
            X_train_t = scaler.fit_transform(X_train_t)
            X_train_v = scaler.transform(X_train_v)
        neg, pos = np.bincount(y_train_t)
        model = create_mlp(
            config, X_train.shape[1], output_bias=np.log([pos / neg])
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["early_stopping"]["patience"],
            restore_best_weights=config["training"]["early_stopping"][
                "restore_best_weights"
            ],
        )
        model.fit(
            X_train_t,
            y_train_t,
            validation_data=(X_train_v, y_train_v),
            epochs=config["training"]["epochs"],
            callbacks=[early_stopping],
            verbose=0,
        )
    else:
        model = create_mlp(
            config, X_train.shape[1], output_bias=np.log10([0.01])
        )
        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
        model.fit(
            X_train,
            y_train,
            epochs=config["training"]["epochs"],
            verbose=0,
        )
    if config["logging"]["save_model"]:
        model.save(config["logging"]["save_path"])
        # should also save associated scaler
        if scaler is not None:
            joblib.dump(
                scaler,
                "".join(config["logging"]["save_path"].split(".")[:-1])
                + "_scaler.pkl",
            )
    return


def test_poem(test_data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Loads model and runs it. Returns DataFrame with predictions column.

    Args:
        test_data (pd.DataFrame): _description_
        config (Dict[str, Any]): _description_

    Returns:
        pd.DataFrame: _description_
    """

    # sometimes we may wish to pass mechanistic outputs with presentation = 0 (e.g. for HLA LOH)
    # we should remove these and set them to zero

    mask = test_data["predicted_pmhc"] > 0

    poem_input = make_poem_input_layer(
        test_data["mhc_i"].values[mask],
        test_data["length"].values[mask],
        test_data["peptide"].values[mask],
        test_data["predicted_pmhc"].values[mask],
        config["dataset"]["length_encoding"],
        config["dataset"]["mhci_encoding"],
        config["dataset"]["mhci_sequence"],
        config["dataset"]["tcr_encoding"],
    )

    version = config["test"]["version"]

    if version == "mlp":
        # load model(s)
        pipeline = load_model(
            os.path.abspath(
                os.path.join(
                    current_dir, "..", "..", config["test"]["mlp_path"]
                )
            )
        )

        scaler = joblib.load(
            os.path.abspath(
                os.path.join(
                    current_dir,
                    "..",
                    "..",
                    "".join(config["test"]["mlp_path"].split(".")[:-1])
                    + "_scaler.pkl",
                )
            )
        )

        poem_input = scaler.transform(poem_input)
        test_data["poem_prediction"] = np.zeros(test_data.shape[0])
        test_data.loc[mask, "poem_prediction"] = pipeline.predict(poem_input)

    elif version == "lgbm":
        pipeline = joblib.load(
            os.path.abspath(
                os.path.join(
                    current_dir, "..", "..", config["test"]["lgbm_path"]
                )
            )
        )
        test_data["poem_prediction"] = np.zeros(test_data.shape[0])
        test_data.loc[mask, "poem_prediction"] = pipeline.predict_proba(
            poem_input
        )[:, 1]

    elif version == "ensemble":
        pipeline_mlp = load_model(
            os.path.abspath(
                os.path.join(
                    current_dir, "..", "..", config["test"]["mlp_path"]
                )
            )
        )

        mlp_scaler = joblib.load(
            os.path.abspath(
                os.path.join(
                    current_dir, "..", "..", config["test"]["mlp_scaler_path"]
                )
            )
        )

        pipeline_logreg = joblib.load(
            os.path.abspath(
                os.path.join(
                    current_dir, "..", "..", config["test"]["logreg_path"]
                )
            )
        )

        logreg_scaler = pickle.load(
            open(
                os.path.abspath(
                    os.path.join(
                        current_dir,
                        "..",
                        "..",
                        config["test"]["logreg_scaler_path"],
                    )
                ),
                "rb",
            )
        )

        pipeline_lgbm = joblib.load(
            os.path.abspath(
                os.path.join(
                    current_dir, "..", "..", config["test"]["lgbm_path"]
                )
            )
        )
        lgbm_scores = pipeline_lgbm.predict_proba(poem_input)[:, 1]
        mlp_scores = pipeline_mlp.predict(
            mlp_scaler.transform(poem_input)
        ).flatten()
        logreg_scores = pipeline_logreg.predict_proba(
            logreg_scaler.transform(poem_input)
        )[:, 1]
        test_data["poem_prediction_mlp"] = np.zeros(test_data.shape[0])
        test_data.loc[mask, "poem_prediction_mlp"] = mlp_scores
        test_data["poem_prediction_lgbm"] = np.zeros(test_data.shape[0])
        test_data.loc[mask, "poem_prediction_lgbm"] = lgbm_scores
        test_data["poem_prediction_logreg"] = np.zeros(test_data.shape[0])
        test_data.loc[mask, "poem_prediction_logreg"] = logreg_scores
        # 0.5 * (lgbm_scores + mlp_scores)

    return test_data


# Functions to check user settings will work
PERMITTED_TCR_ENCODINGS = ["nettepi", "prime", "none"]
PERMITTED_MHCI_SEQUENCE = ["pseudosequence", "full", "none"]
PERMITTED_MHCI_ENCODINGS = [enc.casefold() for enc in VALID_ENCODINGS] + [
    "none]"
]
PERMITTED_NORMALIZATIONS = ["minmax", "standard", "none"]
# TODO: add other tensorflow activation functions to this list
PERMITTED_ACTIVATIONS = ["relu", "sigmoid", "tanh"]
PERMITTED_OPTIMIZERS = ["sgd", "adam"]
PERMITTED_LOSS_FUNCTIONS = ["binary_crossentropy"]


def validate_yaml_string_inputs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and normalizes the YAML to ensure free text user-provided
    options will work.

    Args:
        config (Dict[str, Any]): Loaded config from YAML

    Returns:
        Dict[str, Any]: Processed and validated YAML
    """

    # check dataset options
    for opt, perm in zip(
        ["tcr_encoding", "mhci_sequence", "mhci_encoding", "normalization"],
        [
            PERMITTED_TCR_ENCODINGS,
            PERMITTED_MHCI_SEQUENCE,
            PERMITTED_MHCI_ENCODINGS,
            PERMITTED_NORMALIZATIONS,
        ],
    ):
        # convert to lower case
        config["dataset"][opt] = config["dataset"][opt].casefold()
        # check in permitted list
        if config["dataset"][opt] not in perm:
            raise ValueError(
                f"Invalid choice of {opt}: {config['dataset'][opt]}. "
                f"Must be one of: {', '.join(perm)}"
            )

    # check model options
    for opt, perm in zip(
        ["activation", "output_activation"],
        [
            PERMITTED_ACTIVATIONS,
            PERMITTED_ACTIVATIONS,
        ],
    ):
        # convert to lower case
        config["model"][opt] = config["model"][opt].casefold()
        # check in permitted list
        if config["model"][opt] not in perm:
            raise ValueError(
                f"Invalid choice of {opt}: {config['model'][opt]}. "
                f"Must be one of: {', '.join(perm)}"
            )

    # check training options
    for opt, perm in zip(
        ["optimizer", "loss_function"],
        [
            PERMITTED_OPTIMIZERS,
            PERMITTED_LOSS_FUNCTIONS,
        ],
    ):
        # convert to lower case
        config["training"][opt] = config["training"][opt].casefold()
        # check in permitted list
        if config["training"][opt] not in perm:
            raise ValueError(
                f"Invalid choice of {opt}: {config['training'][opt]}. "
                f"Must be one of: {', '.join(perm)}"
            )

    return config


def validate_yaml_numerical_inputs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validates the YAML to ensure numerical user-provided options will work.

    Args:
        config (Dict[str, Any]): Loaded config from YAML

    Returns:
        Dict[str, Any]: Processed and validated YAML
    """
    # check dropout_rate is a float
    if not isinstance(config["model"]["dropout_rate"], float):
        raise ValueError(
            f"Invalid dropout_rate: {config['model']['dropout_rate']}. "
            "Must be a float."
        )
    # check dropout_rate is in [0.0, 1.0)
    if not (0.0 <= config["model"]["dropout_rate"] < 1.0):
        raise ValueError(
            f"Invalid dropout_rate: {config['model']['dropout_rate']}. "
            "Must be in [0.0, 1.0)"
        )
    # check hidden_layers is either int or a list of ints
    if not isinstance(config["model"]["hidden_layers"], int):
        if not isinstance(config["model"]["hidden_layers"], list):
            raise ValueError(
                f"Invalid hidden_layers: {config['model']['hidden_layers']}. "
                "Must be an int or a list of ints."
            )
        else:
            # still need to check values are ints
            if not all(
                isinstance(i, int) for i in config["model"]["hidden_layers"]
            ):
                raise ValueError(
                    f"Invalid hidden_layers: {config['model']['hidden_layers']}. "
                    "Must be a list of integers."
                )
            else:
                # save number of hidden layers for ease later
                config["model"]["n_hidden_layers"] = len(
                    config["model"]["hidden_layers"]
                )
    # check validation_split is a float
    if not isinstance(config["training"]["validation_split"], float):
        raise ValueError(
            f"Invalid validation_split: {config['training']['validation_split']}. "
            "Must be a float."
        )
    # check validation_split is in [0.0, 1.0)
    if not (0.0 <= config["training"]["validation_split"] < 1.0):
        raise ValueError(
            f"Invalid validation_split: {config['training']['validation_split']}. "
            "Must be in [0.0, 1.0)"
        )
    # check epochs is an int
    if not isinstance(config["training"]["epochs"], int):
        raise ValueError(
            f"Invalid epochs: {config['training']['epochs']}. "
            "Must be an int."
        )
    # check batch_size is an int
    if not isinstance(config["training"]["batch_size"], int):
        raise ValueError(
            f"Invalid batch_size: {config['training']['batch_size']}. "
            "Must be an int."
        )
    # check patience is an int
    if not isinstance(config["training"]["early_stopping"]["patience"], int):
        raise ValueError(
            f"Invalid early stopping patience: {config['training']['early_stopping']['patience']}. "
            "Must be an int."
        )
    # check learning_rate is a float
    if not isinstance(config["training"]["learning_rate"], float):
        raise ValueError(
            f"Invalid learning_rate: {config['training']['learning_rate']}. "
            "Must be a float."
        )
    return config


def fix_random_states(random_seed: int):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


def create_mlp(
    config: Dict[str, Any], input_dim: int, output_bias: np.ndarray
):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    if isinstance(config["model"]["hidden_layers"], int):
        # we have a single hidden layer
        model.add(
            Dense(
                config["model"]["hidden_layers"],
                activation=config["model"]["activation"],
            )
        )
    else:
        hidden_layers = config["model"]["hidden_layers"]
        model.add(
            Dense(
                hidden_layers[0],
                activation=config["model"]["activation"],
            )
        )
        for i in range(1, len(hidden_layers)):
            model.add(
                Dense(
                    hidden_layers[i],
                    activation=config["model"]["activation"],
                )
            )
    # create output layer
    model.add(
        Dense(
            1,
            activation=config["model"]["output_activation"],
            # bias_initializer=tf.keras.initializers.Constant(output_bias),
        )
    )

    # define optimizer
    if config["training"]["optimizer"] == "sgd":
        optimizer = SGD(learning_rate=config["training"]["learning_rate"])
    elif config["training"]["optimizer"] == "adam":
        optimizer = Adam(learning_rate=config["training"]["learning_rate"])

    # compile model
    model.compile(
        optimizer=optimizer,
        loss=config["training"]["loss_function"],
        metrics=["accuracy"],
    )
    return model
