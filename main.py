import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # for numerical stability
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def cross_entropy(y_true, y_pred):
    epsilon = 1e-15 # -inf가 나올 수 있으므로 아주 작은 값 더하기.
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - np.sum(y_true * np.log(y_pred))

# 예시 데이터
y_true = np.array([0, 1, 0])
y_pred = np.array([0.3, 0.6, 0.1])

# Softmax 적용
y_pred_softmax = softmax(y_pred)

# Cross entropy 계산
ce = cross_entropy(y_true, y_pred_softmax)

print("Softmax 적용 결과:", y_pred_softmax)
print("Cross Entropy 값:", ce)

