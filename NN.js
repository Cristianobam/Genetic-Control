class NN {
    constructor(a, b, c, d) {
        if (a instanceof tf.Sequential) {
            this.model = a;
            this.input = b;
            this.hidden = c;
            this.output = d;
        } else {
            this.input = a;
            this.hidden = b;
            this.output = c;
            this.model = this.build();
        }
    }

    build() {
        const model = tf.sequential()

        const input = tf.layers.dense({
            units: this.hidden,
            inputShape: this.input,
            activation: 'sigmoid'
        });

        const output = tf.layers.dense({
            units: this.output,
            activation: 'softmax'
        });

        model.add(input);
        model.add(output);

        return model
    }

    predict(inputs) {
        let xs = tf.tensor2d([inputs])
        return this.model.predict(xs).dataSync()
    }

    copy() {
        return tf.tidy(() => {
            const modelCopy = this.build();
            const weights = this.model.getWeights();
            const copiedWeights = [];

            for (let i = 0; i < weights.length; i++) {
                let tensor = weights[i];
                let shape = tensor.shape;
                let values = tensor.dataSync().slice();

                copiedWeights[i] = tf.tensor(values, shape);
            }

            modelCopy.setWeights(copiedWeights);
            return new NN(modelCopy, this.input, this.hidden, this.output);
        });
    }

    mutate(rate) {
        tf.tidy(() => {
            const weights = this.model.getWeights();
            const mutatedWeights = [];

            for (let i = 0; i < weights.length; i++) {
                let tensor = weights[i];
                let shape = tensor.shape;
                let values = tensor.dataSync().slice();

                for (let j = 0; j < values.length; j++) {
                    if (random(1) < rate) {
                        values[j] += randomGaussian();
                    }
                }

                mutatedWeights[i] = tf.tensor(values, shape);
            }
            this.model.setWeights(mutatedWeights);
        });
    }
}