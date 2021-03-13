import tf from "@tensorflow/tfjs-node-gpu";
export class Pesos {
    constructor() {

        const model = tf.sequential();
        model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
        model.compile({
            optimizer: 'sgd',
            loss: 'meanSquaredError'
        });

        const height = tf.tensor2d([1.82, 1.50, 1.35, 1.75], [4, 1]);
        const weight = tf.tensor2d([92, 60, 45, 85], [4, 1]);


        model.fit(height, weight, {
            epochs: 1000,
            batchSize: 32,
            callbacks: {}
        }).then(info => {
            console.log('Final accuracy', info.history.acc);
            // Predict 3 random samples.
            const prediction = model.predict(tf.tensor2d([1.80], [1, 1]));
            console.log(prediction.toString());
            // model.save('file://modelo_saved'); 
        });

    }
}