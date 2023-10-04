from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import mlflow

import matplotlib.pyplot as plt
import pandas as pd 


class Model:
    def __init__(self) -> None:
        pass

    def load_data(self, path:str):
        df = pd.read_excel(path,parse_dates = ['Time'])  
        df = df.sort_values(by=['Time']).drop('Unnamed: 0', axis=1).set_index('Time')
        df = df.sum(axis=1, numeric_only=True).to_frame(name='SumValue')
        df = df.loc[(df!=0).any(axis=1)]
        return df

    def test_train_split(self, data, random_state:float = 0.3):
        return train_test_split(data['SumValue'][:-1].values.reshape(-1,1),
                                                             data['SumValue'][1:].values.reshape(-1,1),
                                                             test_size=random_state, shuffle=False)

    def train_model(self, X_train, y_train):
        self.model = LinearRegression(n_jobs=-1, positive = True, fit_intercept=False)
        self.model.fit(X_train, y_train)
    
    def test_model(self, test):
        return self.model.predict(test)

    def _plot(self, test_predictions, test):
        plt.figure(figsize = (14, 7))
        plt.title('Result')
        plt.plot(test_predictions, 'r', test, 'g')
        plt.show()

    def print_result(self, test, test_predictions, plot):
        print(f'Test_R2 {r2_score(test,test_predictions)}')
        mlflow.log_metrics({'Test_R2':r2_score(test,test_predictions)})
        if plot:self._plot(test_predictions, test)

    def run(self, path: str, plot:bool = True):
        data = self.load_data(path)
        mlflow.sklearn.autolog()
        X_train, X_test, y_train, _= self.test_train_split(data, random_state=0.3)
        self.train_model(X_train,y_train)
        self.print_result(X_test, self.test_model(X_test), plot)

if __name__ == "__main__":
    Model().run('ML.xlsx', plot=True)


