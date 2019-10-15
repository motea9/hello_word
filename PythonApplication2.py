from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

wine = datasets.load_wine()
print(wine)

wine_data =  wine.data
# 定義資料特徵
wine_target = wine.target
# 定義資料標籤
print(pd.DataFrame(wine.data))
# 印出資料特徵查看
print(pd.DataFrame(wine.target))
# 印出資料標籤查看
x_train, x_test, y_train, y_test = train_test_split(wine_data, wine_target, test_size = 0.2)
# 使用"train_test_spit"將數據分成訓練和測試兩類,test_size = 0.2,代表測試數據佔20%


print('x_test:測試用特徵')
print(x_test)
print('----------------------------------------------------------')
print('x_train:訓練用特徵')
print(x_train)
print('----------------------------------------------------------')
print('y_test:測試用標籤')
print(y_test)
print('----------------------------------------------------------')
print('y_train:訓練用標籤')
print(y_train)
print('----------------------------------------------------------')
print('KNN')
knn=KNeighborsClassifier(p = 1)
knn.fit(x_train, y_train)
print(knn.predict(x_test))
print(y_test)
