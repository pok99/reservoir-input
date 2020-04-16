import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor as gpr



def main():
    x = np.arange(0, 20, 0.1)
    x_list = [[i] for i in x]
    fx = np.sin(x) + 1.3*np.cos(1.1*x+1) - 2.3 * np.sin(.7*x - 1.4) - 30


    interval = 30


    x_filter = x_list[::interval].copy()
    y_filter = fx[::interval].copy()


    x_fit = x_filter
    y_fit = y_filter

    gp = gpr().fit(x_fit, y_fit)


    y_prediction, y_std = gp.predict(x_list, return_std=True)



    print(gp.get_params(deep=True))

    


    plt.plot(x, fx, label='true')
    plt.plot(x, y_prediction, label='prediction', color='red')
    # plt.plot(x, gp.sample_y(x_list, random_state=1), '--', label='s1')
    # plt.plot(x, gp.sample_y(x_list, random_state=2), '--', label='s2')
    # plt.plot(x, gp.sample_y(x_list, random_state=3), '--', label='s3')
    plt.fill_between(x, y_prediction - y_std, y_prediction + y_std, alpha=.3, color='red')
    plt.scatter(x[::interval], y_filter)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()