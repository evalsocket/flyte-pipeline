import os
import re
import string
from typing import Tuple
import flytekit
from flytekit import task, workflow, Secret, Resources
from flytekit.types.directory import FlyteDirectory
from flytekitplugins.kftensorflow import TfJob
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from dataclasses import dataclass
from dataclasses_json import dataclass_json

SECRET_NAME = "user_secret"
SECRET_GROUP = "user-info"
MODEL_FILE_PATH = "saved_model/"

resources = Resources(
    gpu="2", mem="10Gi", storage="10Gi", ephemeral_storage="500Mi"
)


@dataclass_json
@dataclass
class Hyperparameters(object):
    batch_size_per_replica: int = 64
    seed: int = 10000
    epochs: int = 10
    max_features: int = 1000
    sequence_length: int = 4
    embedding_dim: int = 16


class Dataset:
    def __init__(self, train_data: BatchDataset, val_data: BatchDataset, test_data: BatchDataset):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data


def custom_standardization(input_data: str):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(
        stripped_html, '[%s]' % re.escape(string.punctuation), '')


class VectorizedLayer:
    def __init__(self, max_features: int, sequence_length: int):
        self.vectorized_layer = TextVectorization(
            standardize=custom_standardization,
            max_tokens=max_features,
            output_mode='int',
            output_sequence_length=sequence_length)

    def vectorized_text(self, text: str, label: str):
        text = tf.expand_dims(text, -1)
        return self.vectorize_layer(text), label


def get_dataset(data_dir: FlyteDirectory, hyperparameters: Hyperparameters) -> Dataset:
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        os.path.join(os.path.dirname(data_dir.path), 'train'),
        batch_size=hyperparameters.batch_size_per_replica,
        validation_split=0.2,
        subset='training',
        seed=hyperparameters.seed
    )
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        os.path.join(os.path.dirname(data_dir.path), 'train'),
        batch_size=hyperparameters.batch_size_per_replica,
        validation_split=0.2,
        subset='validation',
        seed=hyperparameters.seed
    )
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        str(os.path.join(os.path.dirname(data_dir.path), 'test')),
        batch_size=hyperparameters.batch_size_per_replica
    )

    return Dataset(raw_train_ds, raw_val_ds, raw_test_ds)


@task(
    retries=2,
    cache=True,
    cache_version="1.0",
)
def download_dataset(uri: str) -> Dataset:
    data = tf.keras.utils.get_file(
        "aclImdb_v1",
        uri,
        untar=True,
        cache_dir='.',
        cache_subdir='')
    return FlyteDirectory(path=os.path.join(os.path.dirname(data), "aclImdb_v1"))


@task(
    retries=2,
    cache=True,
    cache_version="1.0",
)
def prepare_dataset(data_dir: FlyteDirectory, hyperparameters: Hyperparameters) -> (VectorizedLayer, Dataset):
    data_set = get_dataset(data_dir=data_dir, hyperparameters=hyperparameters)

    train_text = data_set.train_data.map(lambda x, y: x)
    vectorized_layer = VectorizedLayer(max_features=hyperparameters.max_features,
                                       sequence_length=hyperparameters.sequence_length)
    vectorized_layer.adapt(train_text)

    data_set.train_data = data_set.train_data.map(vectorized_layer.vectorized_text)
    data_set.val_data = data_set.val_data.map(vectorized_layer.vectorized_text)
    data_set.test_data = data_set.test_data.map(vectorized_layer.vectorized_text)

    autotune = tf.data.AUTOTUNE
    data_set.train_data = data_set.train_data.cache().prefetch(buffer_size=autotune)
    data_set.val_data = data_set.val_data.cache().prefetch(buffer_size=autotune)
    data_set.test_data = data_set.test_data.cache().prefetch(buffer_size=autotune)

    return vectorized_layer, data_set


@task(
    task_config=TfJob(num_workers=2, num_ps_replicas=1, num_chief_replicas=1),
    retries=2,
    secret_requests=[Secret(group=SECRET_GROUP, key=SECRET_NAME)],
    cache=True,
    cache_version="1.0",
    requests=resources,
    limits=resources,
)
def create_model(data_set: Dataset, hyperparameters: Hyperparameters, vectorized_layer: VectorizedLayer) \
        -> Tuple[tf.keras.Model, FlyteDirectory]:
    working_dir = flytekit.current_context().working_directory
    checkpoint_dir = "training_checkpoints"
    checkpoint_prefix = os.path.join(working_dir, checkpoint_dir, "ckpt_{epoch}")

    run = neptune.init(
        project="evalsocket/flyte-pipeline",
        api_token=flytekit.current_context().secrets.get(SECRET_GROUP, SECRET_NAME),
    )

    params = {"max_features": hyperparameters.max_features, "optimizer": "Adam", "epochs": hyperparameters.epochs,
              "embedding_dim": hyperparameters.embedding_dim}
    run["parameters"] = params

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True
        ),
        NeptuneCallback(run=run, base_namespace="training"),
    ]

    model = tf.keras.Sequential([
        layers.Embedding(hyperparameters.max_features + 1, hyperparameters.embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)])
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
    _ = model.fit(
        data_set.train_data,
        validation_data=data_set.val_data,
        epochs=hyperparameters.epochs,
        callbacks=callbacks,
    )

    model.save(MODEL_FILE_PATH, save_format="tf")

    export_model = tf.keras.Sequential([
        vectorized_layer,
        model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

    eval_metrics = export_model.evaluate(data_set.test_data)

    for j, metric in enumerate(eval_metrics):
        run["eval/{}".format(model.metrics_names[j])] = metric

    return model, FlyteDirectory(path=os.path.join(working_dir, checkpoint_dir))


@workflow
def train_and_export(uri: str, hyperparameters: Hyperparameters = Hyperparameters()) \
        -> Tuple[tf.keras.Model, FlyteDirectory]:
    data = download_dataset(uri=uri)
    vectorized_layer, data_set = prepare_dataset(
        data_dir=data, hyperparameters=hyperparameters)
    model, checkpoint = create_model(data_set=data_set, hyperparameters=hyperparameters,
                                     vectorized_layer=vectorized_layer)
    return model, checkpoint


if __name__ == "__main__":
    print(f"Running {__file__} main...")
    print(
        f"Running chain_tasks_wf()... {train_and_export(uri='https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')}")
