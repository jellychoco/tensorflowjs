const tf = require("@tensorflow/tfjs");

console.log(tf);

// 1. 과거의 데이터를 준비합니다.
const 온도 = [20, 21, 22, 23];
const 판매량 = [40, 42, 44, 46];
const 원인 = tf.tensor(온도); //텐서화
const 결과 = tf.tensor(판매량); //텐서화
// console.log({ 원인, 결과 });
console.log(원인.print(), 결과.print());

// 2. 모델의 모양을 만듭니다.
const X = tf.input({ shape: [1] });
const Y = tf.layers.dense({ units: 1 }).apply(X);
const model = tf.model({ inputs: X, outputs: Y });
const compileParam = {
  optimizer: tf.train.adam(),
  loss: tf.losses.meanSquaredError, //평균제곱오차
};
// optimizer: 효율적으로 모델을 만드는 방법
// loss : 얼마나 잘 만들어졌는지 측정할때 사용하는 측정방법
model.compile(compileParam);

// 3. 데이터로 모델을 학습시킵니다.
const fitParam = { epochs: 100 };
// const fitParam = {
//   epochs: 100,
//   callbacks: {
//     onEpochEnd: (epoch, logs) => {
//       console.log("epoch", epoch, Math.sqrt(logs.loss)); //평균제곱오차는 실제로 모든값들의 차이보다 훨신 큰값이 나올수있다. 따라서 평균 제곱근 오차로 표현하는게 좋다
//     },
//   },
// }; // loss 추가 예제, loss 가 0에 가까울수록 학습이 잘된것

model.fit(원인, 결과, fitParam).then((result) => {
  // 4. 모델을 이용합니다.
  // 4.1 기존의 데이터를 이용
  const 예측한결과 = model.predict(원인);
  예측한결과.print();
});

// 4.2 새로운 데이터를 이용
// const 다음주온도 = [15,16,17, 18, 19]
// const 다음주원인 = tf.tensor(다음주온도);
// const 다음주결과 = model.predict(다음주원인);
// 다음주결과.print();

const weights = model.getWeights();
const weight = weights[0].arraySync()[0][0];
const bias = weights[1].arraySync()[0];
console.log({ weight, bias });
// console.log(weight[0].arraySync()[0][0]);
