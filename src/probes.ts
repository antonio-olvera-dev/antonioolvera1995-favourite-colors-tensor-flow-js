import tf from "@tensorflow/tfjs-node-gpu";
export class Probes {
    constructor() {
        


        const model = tf.sequential({
            layers: [
                tf.layers.dense({ inputShape: [1], units: 1,activation:"sigmoid" })]
        });
        
        model.compile({
            optimizer: 'sgd',
            loss: 'meanSquaredError',
            metrics: ['accuracy']
         
        });
        
        const x = tf.tensor([1, 2, 3, 4, 5, 6,1, 2, 3, 4, 5, 6,1, 2, 3, 4, 5, 6,1, 2, 3, 4, 5, 6,1, 2, 3, 4, 5, 6]);
        const y = tf.tensor([0, 1, 0, 1, 0, 1,0, 1, 0, 1, 0, 1,0, 1, 0, 1, 0, 1,0, 1, 0, 1, 0, 1,0, 1, 0, 1, 0, 1]);
        
        function onBatchEnd(batch:any, log:any){
            console.log('accuracy', log.acc);
        }
        
        async function predict() {
            try {
        
                const train = await model.fit(x, y, {
                    epochs: 50,
                    batchSize: 1,
                    callbacks: {onBatchEnd}
                });
        
                // console.log('Final accuracy', train.history);
                const prediction = model.predict(tf.tensor([1,2,3,4]));
                console.log(prediction.toString());
        
        
        
            } catch (error) { console.log(error); }
        }
        predict();
        
        
    }
}