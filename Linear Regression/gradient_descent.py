import numpy as np

def gradient_descent(x,y):
    m = 0
    b = 0
    n = len(x)
    iterations = 10000
    learning_rate = 0.08

    for i in range(iterations):
        y_pred = m*x + b
        m_new = (-2/n) * sum(x*(y-y_pred))
        b_new = (-2/n) * sum(y-y_pred)
        cost = (1/n) * sum((y-y_pred)**2)
        m = m - (learning_rate * m_new)
        b = b - (learning_rate * b_new)

        print("Cost is {}, m is {}, b is {}".format(cost,m,b))



x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)













