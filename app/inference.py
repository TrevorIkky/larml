import numpy as np
import tensorflow as tf


class ForestClassificationInference():
    """
    Args:
    model_path = path to tf.saved_model
    Output:
    batch_size, predictions
    Notes: Output of this model is 0/1, adapt for multi-class classification
    """

    def __init__(self, model_path: str) -> None:
        self.model = tf.saved_model.load(model_path)
        print(f'Model signature(s): {self.model.signatures.keys()}')
        self.infer = self.model.signatures["serving_default"]
        print(self.infer.structured_outputs)

    def predict(self, x):
        x = tf.constant(x, tf.float32)
        if len(x.shape) < 4:
            x = tf.expand_dims(x, axis=0)
        x = self.infer(x)
        # Get dict key from self.infer.structured_outputs
        x = np.array(x["output_1"])
        x = (x == 1).astype(int)
        return x


if __name__ == "__main__":
    img = tf.random.uniform([1, 224, 224, 1])
    inference = ForestClassificationInference('./models/forest_classification')
    pred = inference.predict(img)
    print(f'Prediction: {pred}')
