# 모듈 선언
import tensorflow as tf  #모델을 생성 하기 위한 라이브러리
import numpy as np  #: 행렬이나 일반적인 대규모 다차원 배열을 처리하는 라이브러리
from matplotlib import pyplot as plt  #그래프 생성 라이브러리

# 환경설정

# 훈련용 데이터 수 선언
# 학습 데이터를 랜덤하게 생성하여 학습을 하기 위한 훈련용 데이터 수를 지정
trainDataNumber = 100

#모델 최적화를 위한 학습률 선언
#학습률은 학습 모델 수식의 W(Weight)와 b(bias)의 최적의 값을 찾기 위한 최적화 함수의 입력 파라미터임
learningRate = 0.01

#총 학습 선언 횟수
#최적의 모델을 만들기 위한 총 학습 횟수 선언함
totalStep = 1001

#[빌드단계]
# 1. 학습데이터 준비
# 항상 같은 난수를 생성하기 위해 시드 설정
np.random.seed(321)

#학습데이터 리스트 선언
xTrainData = list()
yTrainData = list()

#학습 데이터 생성
xTrainData = np.random.normal(0.0, 1.0, size=trainDataNumber)
for x in xTrainData:
    #y 데이터 생성
    y = 10 * x + 3 + np.random.normal(0.0, 3)
    yTrainData.append(y)

#학습 데이터 확인
plt.plot(xTrainData, yTrainData, 'bo')
plt.title("Train Data")
plt.show

#2. 모델 생성을 위한 변수 초기화
#Weight 변수 선언
w = tf.Variable(tf.random_uniform([1]))
#bias 변수 선언
b = tf.Variable(tf.random_uniform([1]))

#학습 데이터 xTrainDatark가 들어갈 플레이스 홀더 선언
x = tf.placeholder(tf.float32) #독립변수
#학습 데이터 yTrainDatark가 들어갈 플레이스 홀더 선언
y = tf.placeholder(tf.float32) #살제값

#3. 학습 모델 그래프 구성
#3-1) 학습 데이터를 대표하는 가설 그래프 선언
# 방법1 : 일반 연산기호를 이용하여 가설 수식 선언
hypothesis = w * x + b
# 방법2 : 텐서플로우 함수를 이용하여 가설 수식 작성
# hypothesis = tf.add(tf.multiply(w,x),b)

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
#비용함수 그래프를 그리기 위한 변수 선언
weightValueList = list()
costFunctionValueList = list()

print("---------------------------------------------------------------------------------------")
print("Train(Optimization) start")
# totalStep 횟수 만큼 학습
for step in range(totalStep):
    #x,y에 학습 데이터 입력하여 비용함수, w, b, train을 실행
    cost_val, w_val, b_val, _ = sess.run([costFunction, w, b, train], feed_dict={x: xTrainData, y: yTrainData})

    #학습 결과값을 저장
    weightValueList.append(w_val)
    costFunctionValueList.append(cost_val)

    #학습 50회 마다 중간 결과 출력
    if step % 50 == 0:
        print("step : {}, cost : {}, w : {}, b : {}".format(step, cost_val, w_val, b_val))

        #학습 100회 마다 중간 결과 필터링 선 추가
        if step % 100 == 0:
            plt.plot(xTrainData, w_val * xTrainData + b_val, label='Step : {}'.format(step), linewidth=0.5)
print("Train Finished")


print("---------------------------------------------------------------------------------------")
print("[Train result]")
#최적화가 끝난 학습 모델의 비용함수 값
cost_train = sess.run(costFunction, feed_dict={x: xTrainData, y: yTrainData})

#최적화가 끝난 w, b 변수의 값
w_train = sess.run(w)
b_train = sess.run(b)
print("Train cost : {}, w : {}, b : {}".format(cost_train, w_train, b_train))
print("---------------------------------------------------------------------------------------")
print("[Train result]")
#테스트를 위해 x값 선언
testXValue = [2.5]
#최적화된 모델에 x에 대한 y값 계산
resultYValue = sess.run(hypothesis, feed_dict={x: testXValue})
#테스트 결과값 출력
print("x value is {}, y value is {}".format(testXValue, resultYValue))
print("---------------------------------------------------------------------------------------")

#matplotlib를 이용하여 학습 결과 시각화
#결과 확인 그래프
plt.plot(xTrainData, sess.run(w) * xTrainData + sess.run(b), 'r', label='Fitting Line', linewidth=2)
plt.plot(xTrainData, yTrainData, 'bo', label='Train data')
plt.legend()
plt.title("Train Result")
plt.show()

#비용함수 최적화 그래프
plt.plot(weightValueList, costFunctionValueList)
plt.title("costFunction curve")
plt.xlabel("weight")
plt.ylabel("costFunction value")
plt.show()

#세션종료
sess.close()

















