import tf from "@tensorflow/tfjs-node-gpu";
export class Probes {
    constructor() {



        const model = tf.sequential({
            layers: [
                tf.layers.dense({ inputShape: [2], units: 1, activation: "relu", }),
                tf.layers.dense({ units: 1, activation: "sigmoid", }),
            ]
        });

        model.compile({
            optimizer: 'adam',
            loss: 'meanSquaredError',
            metrics: ['binaryAccuracy']


        });

        const x = tf.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
        const y = tf.tensor([[0], [1], [1], [0]]);


        async function predict() {
            try {

                const train = await model.fit(x, y, {
                    epochs: 500,
                    batchSize: 4,
                    callbacks: {}

                });

  
                // console.log('Final accuracy', train.history);
                const prediction = model.predict(tf.tensor([[0, 0], [0, 1], [1, 1]]));
                console.log(prediction.toString());

            } catch (error) { console.log(error); }
        }
        predict(); 


    }
}