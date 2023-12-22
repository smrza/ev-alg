const fs = require('fs');
const path = require('path');

const algorithms = ["DE_best_1_bin", "DE_rand_1_bin", "PSO", "SOMA_all_to_all", "SOMA_all_to_one"];
const functions = ["ackley", "griewank", "levy", "michalewicz", "dixonprice", "perm", "powersum", "rastrigin", "rosenbrock",
    "schwefel", "trid", "nesterov", "alpine_n1", "qing", "salomon", "styblinski", "happy_cat", "quartic",
    "shubert_3", "shubert_4", "shubert", "ackley_n4", "alpine_n2", "xin_she_yang_n2", "xin_she_yang_n4"
];
const dimensions = ["2d", "10d", "30d"]

const resultsPath = path.join(__dirname, 'results');

const data = {};
functions.forEach(func => {
    data[func] = {};
    algorithms.forEach(alg => {
        data[func][alg] = [];
    });
});

algorithms.forEach(algorithm => {
    ['2', '10', '30'].forEach(dimension => {
        functions.forEach(func => {
            const filePath = path.join(resultsPath, algorithm, dimension, `${func}`);

            try {
                const value = parseFloat(fs.readFileSync(filePath, 'utf-8').trim());
                data[func][algorithm].push({ value, dimension });
            } catch (err) {
                console.log(`File not found: ${filePath}`);
            }
        });
    });
});

const functionToDimentionToAlgorithmToValue = {}

functions.forEach(func => {
  functionToDimentionToAlgorithmToValue[func] = {}
  functionToDimentionToAlgorithmToValue[func]["2d"] = []
  functionToDimentionToAlgorithmToValue[func]["10d"] = []
  functionToDimentionToAlgorithmToValue[func]["30d"] = []

  algorithms.forEach(alg => {
    const value2 = data[func][alg][0].value
    const value10 = data[func][alg][1].value
    const value30 = data[func][alg][2].value
    
    const object2d = {}
    object2d[alg] = value2
    functionToDimentionToAlgorithmToValue[func]["2d"].push(object2d)
    const object10d = {}
    object10d[alg] = value10
    functionToDimentionToAlgorithmToValue[func]["10d"].push(object10d)
    const object30d = {}
    object30d[alg] = value30
    functionToDimentionToAlgorithmToValue[func]["30d"].push(object30d)
  })
})


// functions.forEach(func => {
//   console.log(func)
//   dimensions.forEach(dim => {
//     if (dim === "2d") {
//       console.log(functionToDimentionToAlgorithmToValue[func][dim])
//     }
//   })
// })

// functions.forEach(func => {
//   console.log(func)
//   dimensions.forEach(dim => {
//     if (dim === "10d") {
//       console.log(functionToDimentionToAlgorithmToValue[func][dim])
//     }
//   })
// })

functions.forEach(func => {
  console.log(func)
  dimensions.forEach(dim => {
    if (dim === "30d") {
      console.log(functionToDimentionToAlgorithmToValue[func][dim])
    }
  })
})

