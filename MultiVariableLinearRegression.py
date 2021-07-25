# 모듈 선언
import tensorflow as tf  #모델을 생성 하기 위한 라이브러리
import numpy as np  #: 행렬이나 일반적인 대규모 다차원 배열을 처리하는 라이브러리
import matplotlib.pyplot as plt
#3차원 공간에서 그래프 출력
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#[환경설정]
#학습데이터 수 선언
trainDataNumber = 200
#모델 최적화를 위한 학습률 선언
learningRate = 0.01
#총 학습 선언 횟수
totalStep = 1001

#[빌드단계]
# 1. 학습데이터 준비
# 항상 같은 난수를 생성하기 위해 시드 설정
np.random.seed(321)

#학습데이터 리스트 선언
x1TrainData = list()
x2TrainData = list()
yTrainData = list()

#학습 데이터 생성
x1TrainData = np.random.normal(0.0, 1.0, size=trainDataNumber)
x2TrainData = np.random.normal(0.0, 1.0, size=trainDataNumber)

for i in range(0, trainDataNumber):
    #y데이터 생성
    x1 = x1TrainData[i]
    x2 = x2TrainData[i]
    y = 10 * x1 + 5.5 * x2 + 3 + np.random.normal(0.0, 3)
    yTrainData.append(y)

#학습데이터 확인
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1TrainData, x2TrainData, yTrainData, linestyle="none",
        marker="o", mfc="none", markeredgecolor="red")
plt.show()

#[빌드단계]
#2. 모델 생성을 위한 변수 초기화
#Weight 변수 선언
w1 = tf.Variable(tf.random_uniform([1]))
w2 = tf.Variable(tf.random_uniform([1]))
#bias 변수 선언
b = tf.Variable(tf.random_uniform([1]))

#학습 데이터 x1TrainDatark, x2TrainDatark가 들어갈 플레이스 홀더 선언
x1 = tf.placeholder(tf.float32) #독립변수
x2 = tf.placeholder(tf.float32) #독립변수
#학습 데이터 yTrainDatark가 들어갈 플레이스 홀더 선언
y = tf.placeholder(tf.float32) #살제값

#3. 학습 모델 그래프 구성
#3-1) 학습 데이터를 대표하는 가설 그래프 선언
hypothesis = w1 * x1 + w2 * x2 + b

#3-2) 비용함수(오차함수, 손실함수) 선언
costFunction = tf.reduce_mean(tf.square(hypothesis - y))

#3-3) 비용함수의 값이 최소가 되도롤 하는 최적화 함수 선언
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
train = optimizer.minimize(costFunction)


#[실행단계]
#학습 모델 그래프를 실행

#실행을 위한 세션 선언
sess = tf.Session()
#최적화 과정을 통해 구해질 변수 w,b초기화
sess.run(tf.global_variables_initializer())

#학습데이터와 학습 결과를 matplotlib을 이용하여 결과 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1TrainData, x2TrainData, yTrainData,
        linestyle="none", marker="o", mfc="none", markeredgecolor="red")

Xs = np.arange(min(x1TrainData), max(x1TrainData), 0.05)
Ys = np.arange(min(x2TrainData), max(x2TrainData), 0.05)
Xs, Ys = np.meshgrid(Xs, Ys)

print("---------------------------------------------------------------------------------------")
print("Train(Optimization) start")
# totalStep 횟수 만큼 학습
for step in range(totalStep):
    #x,y에 학습 데이터 입력하여 비용함수, w, b, train을 실행
    cost_val, w1_val, w2_val, b_val, _ = sess.run([costFunction, w1, w2, b, train]
                                                  , feed_dict={x1: x1TrainData,
                                                               x2: x2TrainData,
                                                               y: yTrainData})
    #학습 50회 마다 중간 결과 출력
    if step % 50 == 0:
        print("step : {}, cost : {}, w1 : {}, w2 : {}, b : {}".format(step, cost_val, w1_val, w2_val,  b_val))

       #학습 단계 중간 결과 Fitting Surface 추가
        if step % 100 == 0:
            ax.plot_surface(Xs, Ys, w1_val * Xs + w2_val * Ys + b_val,
                            rstride=4, cstride=4, alpha=0.2, cmap=cm.jet)
print("Train Finished")
print("---------------------------------------------------------------------------------------")
#결과확인 그래프
plt.show()
#세션종료
sess.close()





