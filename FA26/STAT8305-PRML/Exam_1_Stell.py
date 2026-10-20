# STAT8305 - Pattern Recognition and Machine Learning
# Exam 1 - Fall 2026
# By Tommy Stell

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder



# Utility functions

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(FILE_PATH, 'STAT8305_next_word_training.csv')
TESTING_DATA_PATH = os.path.join(FILE_PATH, 'STAT8305_next_word_testing_20.csv')
LINEAR_PREDICTIONS_PATH = os.path.join(FILE_PATH, 'Q3_linear_predictions_Stell.csv')
NONLINEAR_PREDICTIONS_PATH = os.path.join(FILE_PATH, 'Q5_nonlinear_predictions_Stell.csv')
SCORES_PATH = os.path.join(FILE_PATH, 'Exam_1_scores_Stell.txt')

SEED = 137
WORD_COLUMNS = ['word1', 'word2', 'word3']

def predict_words(model, X):
    output = model.predict(X)
    if output.ndim == 2:
        return NEXT_WORDS[output.argmax(axis=1)]
    return NEXT_WORDS[output]


def argmax_accuracy(model, X, Y):
    return np.mean(model.predict(X).argmax(axis=1) == Y.argmax(axis=1))


def accuracy_se(accuracy, n):
    return np.sqrt(accuracy * (1 - accuracy) / n)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def save_predictions(predictions, path):
    output = testing_data.copy()
    output['predicted_next_word'] = predictions
    output.to_csv(path, index=False)
    return output


# Question 1

training_data_pre = pd.read_csv(TRAINING_DATA_PATH)
testing_data = pd.read_csv(TESTING_DATA_PATH)

# Shuffle once, then hold out the last 500 rows as the validation set
shuffled = training_data_pre.sample(frac=1, random_state=SEED).reset_index(drop=True)
training_data = shuffled.iloc[:5500]
validation_data = shuffled.iloc[5500:]

# data style := (Word 1, Word 2, Word 3) -> (Next Word)

unique_words_1 = sorted(training_data_pre['word1'].unique())
unique_words_2 = sorted(training_data_pre['word2'].unique())
unique_words_3 = sorted(training_data_pre['word3'].unique())
unique_words_next = sorted(training_data_pre['next_word'].unique())

# One hot encoding for the words. Each word position gets its own block of indicator columns,
# so X = [word1 block | word2 block | word3 block]. The encoders are fit on the full training
# file so the column layout matches the R model.matrix() levels.

encoder_X = OneHotEncoder(categories=[unique_words_1, unique_words_2, unique_words_3],
                          sparse_output=False, handle_unknown='ignore')
encoder_next = OneHotEncoder(categories=[unique_words_next], sparse_output=False)

encoder_X.fit(training_data_pre[WORD_COLUMNS])
encoder_next.fit(training_data_pre[['next_word']])

NEXT_WORDS = encoder_next.categories_[0]

# Column ranges of each word block inside X
block_sizes = [len(c) for c in encoder_X.categories_]
block_starts = np.cumsum([0] + block_sizes)
WORD1_COLS = np.arange(block_starts[0], block_starts[1])
WORD2_COLS = np.arange(block_starts[1], block_starts[2])
WORD3_COLS = np.arange(block_starts[2], block_starts[3])


def encode_X(df):
    return encoder_X.transform(df[WORD_COLUMNS])


def encode_Y(df):
    return encoder_next.transform(df[['next_word']])


X_train = encode_X(training_data)
Y_train = encode_Y(training_data)

X_val = encode_X(validation_data)
Y_val = encode_Y(validation_data)

X_train_full = encode_X(training_data_pre)
Y_train_full = encode_Y(training_data_pre)

X_test = encode_X(testing_data)


def one_hot_preview(n_rows=10):
    # First n rows of the original training file next to their one-hot rows
    X_head = pd.DataFrame(encode_X(training_data_pre.head(n_rows)).astype(int),
                          columns=encoder_X.get_feature_names_out(WORD_COLUMNS))
    return training_data_pre.head(n_rows), X_head


# Question 2

# Two reasonable models will be all three words using linear regression and the other will be
# linear regression with only the last two words. Each is a separate 0-1 regression per possible
# next word (the 12 columns of Y), and the predicted word is the one with the largest fitted value.

def select_columns(X, cols):
    return X[:, cols]


def column_linear_model(cols):
    # Every model takes the full X, the column selector picks out the words it uses
    return make_pipeline(FunctionTransformer(select_columns, kw_args={'cols': cols}),
                         LinearRegression())


Linear_model_1 = column_linear_model(np.concatenate([WORD1_COLS, WORD2_COLS, WORD3_COLS]))
Linear_model_2 = column_linear_model(np.concatenate([WORD2_COLS, WORD3_COLS]))

LINEAR_MODELS = {
    'LM1: word1 + word2 + word3': Linear_model_1,
    'LM2: word2 + word3': Linear_model_2,
}

def linear_fit(LM1, LM2):

    LM1.fit(X_train, Y_train)
    LM2.fit(X_train, Y_train)

    return LM1, LM2


# Question 3

# The selection criterion is validation accuracy of the argmax prediction. K-fold cross-validation
# on the 5500 training rows is reported alongside as a check on the 500-row validation estimate.

K_folds = 10

def linear_score(models):
    folds = KFold(n_splits=K_folds, shuffle=True, random_state=SEED)
    cv_scores = {}
    cv_times = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        scores = cross_val_score(clone(model), X_train, Y_train, cv=folds, scoring=argmax_accuracy)
        t1 = time.perf_counter()
        cv_scores[name] = (scores.mean(), scores.std())
        cv_times[name] = t1 - t0

    return cv_scores, cv_times

def linear_validation_score(models):
    val_scores = {}
    for name, model in models.items():
        model.fit(X_train, Y_train)
        val_scores[name] = argmax_accuracy(model, X_val, Y_val)

    best_name = max(val_scores, key=val_scores.get)

    return best_name, val_scores

def best_linear_model_predicition(best_linear_model):

    # Refit the selected specification on all 6000 training rows before predicting the test set
    final_model = clone(best_linear_model).fit(X_train_full, Y_train_full)

    return predict_words(final_model, X_test)

# Question 4

# two nonlinear models for comparison: 1.) is an extra random forest model with a max depth of 3
# and 2.) is a simple MLP model with 1 hidden layer only.
# Integer labels, since MLPClassifier early stopping fails on string labels in sklearn 1.7.

y_train = Y_train.argmax(axis=1)
y_val = Y_val.argmax(axis=1)
y_train_full = Y_train_full.argmax(axis=1)

NONLINEAR_MODELS = {
    'Extra Random Forest (max_depth=5)': RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=SEED),
    'Simple MLP (1 hidden layer)': MLPClassifier(
        hidden_layer_sizes=(8,), activation='relu', solver='adam', alpha=1e-3,
        max_iter=2000, early_stopping=True, random_state=SEED),
}

def nonlinear_fit(models):
    for model in models.values():
        model.fit(X_train, y_train)

    return models

# Question 5

def nonlinear_cv_score(models):
    folds = KFold(n_splits=K_folds, shuffle=True, random_state=SEED)
    cv_scores = {}
    cv_times = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        scores = cross_val_score(clone(model), X_train, y_train, cv=folds)
        t1 = time.perf_counter()
        cv_scores[name] = (scores.mean(), scores.std())
        cv_times[name] = t1 - t0

    return cv_scores, cv_times

def nonlinear_score(models):
    val_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_scores[name] = np.mean(model.predict(X_val) == y_val)

    best_name = max(val_scores, key=val_scores.get)

    return best_name, val_scores

def best_nonlinear_model_prediction(best_NL_model):

    # Same refit rule as the linear model: all 6000 training rows
    final_model = clone(best_NL_model).fit(X_train_full, y_train_full)

    return predict_words(final_model, X_test)

if __name__ == "__main__":
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)
    n_val = len(validation_data)

    scores_file = open(SCORES_PATH, 'w', encoding='utf-8')
    terminal = sys.stdout
    sys.stdout = Tee(terminal, scores_file)

    # Question 1
    print("Question 1: One-hot encoding")
    print(f"  Levels per word: word1={block_sizes[0]}, word2={block_sizes[1]}, word3={block_sizes[2]}, "
          f"next_word={len(NEXT_WORDS)}")
    print(f"  Full training design matrix: {X_train_full.shape}")
    print(f"  Split: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")
    print("  (R's model.matrix(~ word1 + word2 + word3 - 1) drops the reference level of word2 and "
          f"word3, giving {X_train_full.shape[0]} x {sum(block_sizes) - 2}.)")
    head_original, head_encoded = one_hot_preview()
    print(head_original.to_string(index=False))
    print(head_encoded.to_string(index=False))

    # Question 2
    print("\nQuestion 2: Linear models (one 0-1 regression per next word, predict the argmax)")
    linear_fit_times = {}
    for name, model in LINEAR_MODELS.items():
        t0 = time.perf_counter()
        model.fit(X_train, Y_train)
        t1 = time.perf_counter()
        linear_fit_times[name] = t1 - t0
        print(f"  {name}: training accuracy {argmax_accuracy(model, X_train, Y_train):.3f} (fit time: {(t1 - t0)*1000:.2f} ms)")

    # Question 3
    print("\nQuestion 3: Linear-model selection")
    cv_scores, cv_times_lin = linear_score(LINEAR_MODELS)
    best_linear_name, linear_val_scores = linear_validation_score(LINEAR_MODELS)
    for name in LINEAR_MODELS:
        acc = linear_val_scores[name]
        cv_mean, cv_std = cv_scores[name]
        cv_time = cv_times_lin[name]
        print(f"  {name}: validation accuracy {acc:.3f} (SE {accuracy_se(acc, n_val):.3f}), "
              f"{K_folds}-fold CV {cv_mean:.3f} +/- {cv_std:.3f} (CV time: {cv_time:.3f} s)")
    print(f"  Selected: {best_linear_name}")
    linear_predictions = best_linear_model_predicition(LINEAR_MODELS[best_linear_name])
    save_predictions(linear_predictions, LINEAR_PREDICTIONS_PATH)
    print(f"  Test predictions written to {os.path.basename(LINEAR_PREDICTIONS_PATH)}")

    # Question 4
    print("\nQuestion 4: Nonlinear models")
    nonlinear_fit_times = {}
    for name, model in NONLINEAR_MODELS.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t1 = time.perf_counter()
        nonlinear_fit_times[name] = t1 - t0
        print(f"  {name}: training accuracy {np.mean(model.predict(X_train) == y_train):.3f} (fit time: {(t1 - t0)*1000:.2f} ms)")

    # Question 5
    print("\nQuestion 5: Nonlinear-model selection")
    cv_scores_nl, cv_times_nl = nonlinear_cv_score(NONLINEAR_MODELS)
    best_nonlinear_name, nonlinear_val_scores = nonlinear_score(NONLINEAR_MODELS)
    for name, acc in nonlinear_val_scores.items():
        cv_mean, cv_std = cv_scores_nl[name]
        cv_time = cv_times_nl[name]
        print(f"  {name}: validation accuracy {acc:.3f} (SE {accuracy_se(acc, n_val):.3f}), "
              f"{K_folds}-fold CV {cv_mean:.3f} +/- {cv_std:.3f} (CV time: {cv_time:.3f} s)")
    print(f"  Selected: {best_nonlinear_name}")
    nonlinear_predictions = best_nonlinear_model_prediction(NONLINEAR_MODELS[best_nonlinear_name])
    save_predictions(nonlinear_predictions, NONLINEAR_PREDICTIONS_PATH)
    print(f"  Test predictions written to {os.path.basename(NONLINEAR_PREDICTIONS_PATH)}")

    print("\nLinear vs nonlinear")
    print(f"  Validation accuracy: linear {linear_val_scores[best_linear_name]:.3f}, "
          f"nonlinear {nonlinear_val_scores[best_nonlinear_name]:.3f}")
    comparison = testing_data.copy()
    comparison['linear'] = linear_predictions
    comparison['nonlinear'] = nonlinear_predictions
    print(f"  Test predictions agree on {np.sum(linear_predictions == nonlinear_predictions)} "
          f"of {len(testing_data)} observations")
    print(comparison.to_string(index=False))

    print("\nModel Training & Computation Times Summary")
    t_base = linear_fit_times[list(LINEAR_MODELS.keys())[0]]
    timing_rows = []
    for name, model in LINEAR_MODELS.items():
        t0 = time.perf_counter()
        clone(model).fit(X_train_full, Y_train_full)
        t_full = time.perf_counter() - t0
        rel_time = linear_fit_times[name] / t_base
        timing_rows.append({
            'Model': name,
            'Fit X_train (5,500)': f"{linear_fit_times[name]*1000:.2f} ms",
            'Rel. Training Time': "1.00x (Baseline)" if name == list(LINEAR_MODELS.keys())[0] else f"{rel_time:.2f}x",
            '10-Fold CV (5,500)': f"{cv_times_lin[name]:.3f} s",
            'Refit Full (6,000)': f"{t_full*1000:.2f} ms"
        })
    for name, model in NONLINEAR_MODELS.items():
        t0 = time.perf_counter()
        clone(model).fit(X_train_full, y_train_full)
        t_full = time.perf_counter() - t0
        rel_time = nonlinear_fit_times[name] / t_base
        timing_rows.append({
            'Model': name,
            'Fit X_train (5,500)': f"{nonlinear_fit_times[name]*1000:.2f} ms",
            'Rel. Training Time': f"{rel_time:.1f}x",
            '10-Fold CV (5,500)': f"{cv_times_nl[name]:.3f} s",
            'Refit Full (6,000)': f"{t_full*1000:.2f} ms"
        })
    print(pd.DataFrame(timing_rows).to_string(index=False))

    sys.stdout = terminal
    scores_file.close()
    print(f"\nScores written to {os.path.basename(SCORES_PATH)}")
