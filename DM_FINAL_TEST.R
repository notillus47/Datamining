######################################################################################################################################################

# 컴퓨터공학부 20161553 박현영 데이터마이닝 기말고사 대체 과제

###################################################################### 1번 문제 ######################################################################
# 학습 데이터
x = c(3.0, 6.0, 3.0, 6.0, 7.5, 7.5, 15.0)
u = c(10.0, 10.0, 20.0, 20.0, 5.0, 10.0, 12.0)
y = c(4.56, 5.9, 6.7, 8.02, 7.7, 8.1, 6.1)

# 시각화
library(scatterplot3d)
scatterplot3d(x, u, y, xlim = 2:16, ylim = 4:21, zlim = 0:10, type = 'h')

# 테스트 데이터
test_x = c(7.5, 5.0)
test_u = c(5.0, 12.0)
test_df = data.frame(x = test_x, u = test_u)

# 선형 모델 생성
m = lm(y~x + u)
coef(m)

# 시각화
s = scatterplot3d(x, u, y, xlim = 2:16, ylim = 4:21, zlim = 0:10, type = 'h')
s$plane3d(m)

# 모델 'm'으로 예측 수행     # 6.422700, 6.591978
test_y = predict(m, test_df)
test_y

# 예측한 결과 시각화
s1 = scatterplot3d(test_x, test_u, test_y,
                   xlim = 2:16, ylim = 4:21, zlim = 0:10,
                   pch = 20, type = 'h', color = 'red')
s1$plane3d(m)

######################################################################################################################################################

###################################################################### 5번 문제 ######################################################################

# 1. 데이터를 획득하고 모델을 적용하기 위한 준비과정
# 필요 라이브러리 부착
library(caret)
library(randomForest)
library(rpart)
library(class)
library(e1071)

# 데이터 읽어오기 및 Factor형 변환
ucla = read.csv('https://stats.idre.ucla.edu/stat/data/binary.csv')
ucla$admit = factor(ucla$admit)


# 2. 학습 데이터와 테스트 데이터 분리 과정
# 학습 데이터 및 테스트 데이터 분리 과정
# 6:4로 나누어 6은 학습 데이터, 4는 테스트 데이터로 사용
n = nrow(ucla)
u = 1:n
train_data = sample(u, n*0.6)
test_data = setdiff(u, train_data)
ucla_train_data = ucla[train_data, ]
ucla_test_data = ucla[test_data, ]


# 3. 학습데이터로 모델을 만드는 과정
# 학습 데이터로 모델을 만드는 과정(모델링)
# 결정트리
dt = rpart(admit~., data = ucla_train_data)

# 랜덤포레스트(ntree = 50)
rf_50 = randomForest(admit~., data = ucla_train_data, ntree = 50)

# 랜덤포레스트(ntree = 1000)
rf_1000 = randomForest(admit~., data = ucla_train_data, ntree = 1000)

# knn
knn = train(admit~., data = ucla_train_data, method = 'knn')

# svm(radial basis)
s = svm(admit~., data = ucla_train_data)

# svm(polynomial)
sp = svm(admit~., data = ucla_train_data, kernel = 'polynomial')


# 4. 테스트 데이터로 예측하고, 예측 결과를 혼동행렬로 출력하는 과정
# 결정 트리 예측 결과 및 혼동행렬 출력
pred_dt = predict(dt, ucla_test_data, type = 'class')
table(pred_dt, ucla_test_data$admit)

# 랜덤포레스트 예측 결과 및 혼동행렬 출력(ntree = 50)
pred_rf_50 = predict(rf_50, newdata = ucla_test_data)
table(pred_rf_50, ucla_test_data$admit)

# 랜덤포레스트 예측 결과 및 혼동행렬 출력(ntree = 1000)
pred_rf_1000 = predict(rf_1000, newdata = ucla_test_data)
table(pred_rf_1000, ucla_test_data$admit)

# knn 모델 예측 결과 및 혼동행렬 출력
pred_knn = predict(knn, ucla_test_data)
table(pred_knn, ucla_test_data$admit)

# SVM 모델 예측 결과 및 혼동행렬 출력 (Radial basis)
pred_s = predict(s, ucla_test_data)
table(pred_s, ucla_test_data$admit)

# SVM 모델 예측 결과 및 혼동행렬 출력 (Polynomial)
pred_sp = predict(sp, ucla_test_data)
table(pred_sp, ucla_test_data$admit)


