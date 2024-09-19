import mlflow.keras
import numpy as np
import argparse
from tensorflow.keras.datasets import cifar10

mlflow.set_tracking_uri("http://127.0.0.1:5012")


# 保存されたMLflowモデルをロード
def load_model(run_id):
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.keras.load_model(model_uri)
    return model


# 推論の実行
def run_inference(model, data):
    predictions = model.predict(data)
    return predictions


if __name__ == "__main__":
    # コマンドライン引数を処理するためのargparse
    parser = argparse.ArgumentParser(
        description="Run inference using a saved MLflow model."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="The RUN ID of the saved model in MLflow.",
    )
    args = parser.parse_args()

    # モデルのロード
    model = load_model(args.run_id)

    # CIFAR-10テストデータをロード（例として使用）
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = x_test / 255.0  # データを0-1にスケール

    # テストデータの一部を使って推論
    x_sample = x_test[:5]  # 最初の5つの画像を使用
    predictions = run_inference(model, x_sample)

    # 結果を表示
    print("Predictions:", np.argmax(predictions, axis=1))  # 推論結果（クラスラベル）
    print("True labels:", y_test[:5].flatten())  # 実際のラベル
