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

const functionToAlgToRankByD = {}

functions.forEach(func => {
  functionToAlgToRankByD[func] = []
  dimensions.forEach(dim => {
    const arrayObjects = functionToDimentionToAlgorithmToValue[func][dim]
    arrayObjects.sort((a, b) => {
      const valueA = Object.values(a)[0];
      const valueB = Object.values(b)[0];
      return valueA - valueB;
    });
    functionToAlgToRankByD[func].push(arrayObjects)
  })
})

function calculateRanking(arr, key) {
  let ranksAccrued = 0
  for (let i = 0; i < arr.length; i++) {
    for (let j = 0; j < arr[i].length; j++) {
      const item = arr[i][j];
      const keys = Object.keys(item)
      if (keys[0] === key) {
        ranksAccrued += j+1
      }
    }
  }

  const returningObj = {}
  returningObj[key] = ranksAccrued / 3
  return returningObj;
}

const ranking = {}
functions.forEach(func => {
  ranking[func] = []
  const ranking1 = calculateRanking(functionToAlgToRankByD[func], "DE_best_1_bin")
  const ranking2 = calculateRanking(functionToAlgToRankByD[func], "DE_rand_1_bin")
  const ranking3 = calculateRanking(functionToAlgToRankByD[func], "PSO")
  const ranking4 = calculateRanking(functionToAlgToRankByD[func], "SOMA_all_to_all")
  const ranking5 = calculateRanking(functionToAlgToRankByD[func], "SOMA_all_to_one")
  ranking[func].push(ranking1, ranking2, ranking3, ranking4, ranking5)
})

const columns = Array.from(
  new Set(Object.values(ranking).flatMap(item => item.map(innerItem => Object.keys(innerItem)[0])))
);

const rows = Object.keys(ranking);

const tableHeader = ["", ...columns].join("\t");

const tableRows = rows.map(row => {
  const rowData = [row];
  columns.forEach(column => {
    const value = ranking[row].find(innerItem => Object.keys(innerItem)[0] === column);
    rowData.push(value ? value[column].toFixed(2) : 0);
  });
  return rowData.join("\t");
});

const tableString = [tableHeader, ...tableRows].join("\n");

console.log(tableString);
