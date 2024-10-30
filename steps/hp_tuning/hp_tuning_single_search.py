# MIT License
# 
# Copyright (c) Kevin Hickey 2024
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 


from typing import Any, Dict

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step
from zenml.logger import get_logger

from utils import get_model_from_config

logger = get_logger(__name__)


@step
def hp_tuning_single_search(
    model_package: str,
    model_class: str,
    search_grid: Dict[str, Any],
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    target: str,
) -> Annotated[ClassifierMixin, "hp_result"]:
    """Evaluate a trained model.

    This is an example of a model hyperparameter tuning step that takes
    in train and test datasets to perform a randomized search for best model
    in configured space.

    This step is parameterized to configure the step independently of the step code,
    before running it in a pipeline. In this example, the step can be configured
    to use different input datasets and also have a flag to fall back to default
    model architecture. See the documentation for more information:

        https://docs.zenml.io/how-to/build-pipelines/use-pipeline-step-parameters

    Args:
        model_package: The package containing the model to use for hyperparameter tuning.
        model_class: The class of the model to use for hyperparameter tuning.
        search_grid: The hyperparameter search space.
        dataset_trn: The train dataset.
        dataset_tst: The test dataset.
        target: Name of target columns in dataset.

    Returns:
        The best possible model for given config.
    """
    model_class = get_model_from_config(model_package, model_class)

    for search_key in search_grid:
        if "range" in search_grid[search_key]:
            search_grid[search_key] = range(
                search_grid[search_key]["range"]["start"],
                search_grid[search_key]["range"]["end"],
                search_grid[search_key]["range"].get("step", 1),
            )

    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###

    X_trn = dataset_trn.drop(columns=[target])
    y_trn = dataset_trn[target]
    X_tst = dataset_tst.drop(columns=[target])
    y_tst = dataset_tst[target]
    logger.info("Running Hyperparameter tuning...")
    cv = RandomizedSearchCV(
        estimator=model_class(),
        param_distributions=search_grid,
        cv=3,
        n_jobs=-1,
        n_iter=10,
        random_state=42,
        scoring="accuracy",
        refit=True,
    )
    cv.fit(X=X_trn, y=y_trn)
    y_pred = cv.predict(X_tst)
    score = accuracy_score(y_tst, y_pred)
    # log score along with output artifact as metadata
    log_artifact_metadata(
        metadata={"metric": float(score)},
        artifact_name="hp_result",
    )
    ### YOUR CODE ENDS HERE ###
    return cv.best_estimator_
