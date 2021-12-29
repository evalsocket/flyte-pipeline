import os
import re
import string
import pathlib
import flytekit
from flytekit import task, workflow
from flytekit.types.directory import FlyteDirectory
from flytekit.remote import FlyteRemote
from flytekit.models.core.execution import WorkflowExecutionPhase
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


class Dataset:
    def __init__(self, train_data: BatchDataset, val_data: BatchDataset, test_data: BatchDataset):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data


@task
def download_model(uri: str) -> FlyteDirectory:
    data = tf.keras.utils.get_file(
        "aclImdb_v1",
        uri,
        untar=True,
        cache_dir='.',
        cache_subdir='')
    working_dir = flytekit.current_context().working_directory
    return FlyteDirectory(path=os.path.join(working_dir, "aclImdb_v1"))


@task
def get_dataset(data_dir: FlyteDirectory, batch_size: int, seed: int) -> Dataset:
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        os.path.join(os.path.dirname(data_dir.path), 'train'),
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed
    )
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        os.path.join(os.path.dirname(data_dir.path), 'train'),
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed
    )
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        str(os.path.join(os.path.dirname(data_dir.path), 'test')),
        batch_size=batch_size
    )

    return Dataset(raw_train_ds, raw_val_ds, raw_test_ds)


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


@task
def prepare_dataset(data_set: Dataset, max_features: int, sequence_length: int) -> VectorizedLayer:
    train_text = data_set.train_data.map(lambda x, y: x)
    vectorized_layer = VectorizedLayer(max_features=max_features, sequence_length=sequence_length)
    vectorized_layer.adapt(train_text)

    data_set.train_data = data_set.train_data.map(vectorized_layer.vectorized_text)
    data_set.val_data = data_set.val_data.map(vectorized_layer.vectorized_text)
    data_set.test_data = data_set.test_data.map(vectorized_layer.vectorized_text)

    autotune = tf.data.AUTOTUNE
    data_set.train_data = data_set.train_data.cache().prefetch(buffer_size=autotune)
    data_set.val_data = data_set.val_data.cache().prefetch(buffer_size=autotune)
    data_set.test_data = data_set.test_data.cache().prefetch(buffer_size=autotune)

    return vectorized_layer


@task
def create_model(data_set: Dataset, epochs: int, max_features: int, vectorized_layer: VectorizedLayer) \
        -> (FlyteDirectory, float, float):
    embedding_dim = 16
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
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
        epochs=epochs)

    working_dir = flytekit.current_context().working_directory
    pp = pathlib.Path(os.path.join(working_dir, "model"))
    pp.mkdir(exist_ok=True)

    export_model = tf.keras.Sequential([
        vectorized_layer,
        model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy'])

    loss, accuracy = export_model.evaluate(data_set.test_data)

    return FlyteDirectory(path=os.path.join(working_dir, "model")), loss, accuracy

@task
def server_model_server(domain: str, project: str) \
        # TODO: Add logic fro getting output of a execution and use it for serving
        remote = FlyteRemote.from_config(
            default_project=project,
            default_domain=domain,
            config_file_path="./config",
        )


@workflow
def train_and_export(uri: str, batch_size: int, seed: int, max_features: int, sequence_length: int, epochs: int) \
        -> (FlyteDirectory, float, float):
    data = download_model(uri=uri)
    data_set = get_dataset(data_dir=data, batch_size=batch_size, seed=seed)
    vectorized_layer = prepare_dataset(
        data_set=data_set, max_features=max_features, sequence_length=sequence_length)
    model, lost, accuracy = create_model(data_set=data_set, epochs=epochs, max_features=max_features,
                                         vectorized_layer=vectorized_layer)
    return model, lost, accuracy

@workflow
def serve():
    server_model_server()

if __name__ == "__main__":
    print(f"Running {__file__} main...")
    print(
        f"Running chain_tasks_wf()... {train_and_export(uri='https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', batch_size=5, seed=7)}")
