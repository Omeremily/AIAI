import { Data } from "./data/dataTypes";

declare const brain: any;

// Define candy input and output types
interface CandyInput {
    sweetness: number;
    crunchiness: number;
    flavorIntensity: number;
}

interface CandyOutput {
    candyType: string;
}

// Encode function to convert string to numeric representation
function encode(arg: string): number[] {
    return arg.split('').map(x => (x.charCodeAt(0) / 256));
}

// Normalize candy input values
function normalizeCandyInput(input: CandyInput): CandyInput {
    return {
        sweetness: input.sweetness / 100,
        crunchiness: input.crunchiness / 10,
        flavorIntensity: input.flavorIntensity / 5
    };
}

// Create and train candy prediction neural network
export function createCandyNN() {
    const net = new brain.NeuralNetwork({
        hiddenLayers: [3],
        inputSize: 3,
        outputSize: 1
    });

    // Define candy training data
    const trainingDataCandy: Array<Data<CandyInput, CandyOutput>> = [
        { input: { sweetness: 80, crunchiness: 7, flavorIntensity: 4 }, output: { candyType: "Chocolate" } },
        { input: { sweetness: 50, crunchiness: 9, flavorIntensity: 3 }, output: { candyType: "Gummy Bears" } },
        { input: { sweetness: 70, crunchiness: 5, flavorIntensity: 6 }, output: { candyType: "Lollipop" } },
        { input: { sweetness: 60, crunchiness: 8, flavorIntensity: 4 }, output: { candyType: "Jelly Beans" } },
        { input: { sweetness: 90, crunchiness: 3, flavorIntensity: 7 }, output: { candyType: "Caramel" } },
        { input: { sweetness: 65, crunchiness: 6, flavorIntensity: 5 }, output: { candyType: "Hard Candy" } },
        { input: { sweetness: 85, crunchiness: 4, flavorIntensity: 8 }, output: { candyType: "Licorice" } },
        { input: { sweetness: 55, crunchiness: 7, flavorIntensity: 3 }, output: { candyType: "Toffee" } },
        { input: { sweetness: 75, crunchiness: 5, flavorIntensity: 6 }, output: { candyType: "Fudge" } },
        { input: { sweetness: 70, crunchiness: 6, flavorIntensity: 7 }, output: { candyType: "Jawbreaker" } },
    ];

    // Process and encode candy training data
    const processedTrainingData = trainingDataCandy.map(data => ({
        input: encode(JSON.stringify(normalizeCandyInput(data.input))),
        output: data.output
    }));

    // Train neural network
    net.train(processedTrainingData, {
        logPeriod: 100,
        log: (stats: any) => console.log(stats)
    });

    // Make candy predictions
    const newCandyInput: CandyInput = { sweetness: 75, crunchiness: 8, flavorIntensity: 4 };
    const encodedInput = encode(JSON.stringify(normalizeCandyInput(newCandyInput)));
    const candyPrediction = net.run(encodedInput);

    console.log('Candy Prediction:', candyPrediction); // Output candy prediction

    return net; // Return trained neural network
}





/////////////רשת LSTM


//ייבוא דאטה

import {candy} from './data/candy';

//סידור הדאטה
let trainingData:string[] = candy.split('. '); //יצירת מערך משפטים
trainingData = trainingData.map((sentence)=>{ // מחיקת תווים מסויימים
    return sentence.replace('\n', '');
})

console.log(trainingData); // בדיקה שנראה טוב

//יצירת מכונה
let net = new brain.recurrent.LSTM({
    hiddenLayers: [4,8,8],
}); 

//אימון מכונה
net.train(trainingData, {
    iterations: 1000, //אלף סיבובים
    errorTresh: 0.01, //שגיאה מינימלית 
    logPeriod: 100, //להראות פעם ב100
    log: (stats: any) => console.log(stats) // סטטיסטיקה של האימון
});


//הצגת רשת LSTM

let div = document.querySelector('#app') as HTMLDivElement;
div.innerHTML= brain.utilities.toSVG(net);


//בדיקה
console.log(net.run('what is a small bean-shaped candy?'));