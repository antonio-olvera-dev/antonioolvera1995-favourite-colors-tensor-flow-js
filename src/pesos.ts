import tf from "@tensorflow/tfjs-node-gpu";
export class Pesos {
    constructor() {

        const model = tf.sequential();
        model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
        model.compile({
            optimizer: 'sgd',
            loss: 'meanSquaredError'
        });

        const height = tf.tensor([1.82, 1.50, 1.35, 1.75, 1.82, 1.50, 1.35, 1.75], [8, 1]);
        const weight = tf.tensor([92, 60, 45, 85, 92, 60, 45, 85], [8, 1]);


        model.fit(height, weight, {
            epochs: 15000,
            batchSize: 32,
            callbacks: {}
        }).then(info => {
            console.log('Final accuracy', info.history.acc);
            // Predict 3 random samples.
            const prediction = model.predict(tf.tensor([1.80], [1, 1]));
            console.log(prediction.toString());
            // model.save('file://modelo_saved'); 
        });

    }
}