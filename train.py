import os
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

from model import create_keras
from datasets import load_cifar10, save


# モデルの学習
def train_model():
    model = create_keras()
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Experimentの設定（存在しない場合は新規作成、存在する場合は使用）
    mlflow.set_experiment("cifar10_experiment")

    # CIFAR-10データセットをロード
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    # MLflowのトラッキングを開始
    with mlflow.start_run():
        # データセットを保存
        dataset_filename = "cifar10_train_data.npz"
        save(x_train, y_train, dataset_filename)
        mlflow.log_artifact(dataset_filename)

        # モデルの学習
        model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

        # モデルの評価
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

        # パラメータ、メトリクス、モデルを記録
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("epochs", 1)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.keras.log_model(model, "model")

        os.remove(dataset_filename)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5012")
    train_model()
