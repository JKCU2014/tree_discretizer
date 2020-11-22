# 基于Sklearn各类树模型的特征工程工具

### Author: 何君柯
### E-mail: zachcu2015@yahoo.com

+ 基于sklearn的决策树分类器、GBDT算法的离散化工具，具有和sklearn中实现算法类似的interface，可作为参数传入GridSearchCV、Pipeline等。
+ 工程通过提取决策树中各节点信息，可用于对单变量做离散化或生成新的交叉变量。
+ 该工程主要实现了对各节点路径提取后的名称做了记录，相对于直接使用GBDTClassifier的apply函数，满足了对于模型解释性的需求。
+ 导入pkg/dtree\_discretizer.py中的对象即可使用该工程
+ dtree\_discretizer.py中目前可用对象：ClassificationDTDiscretizer、RegressionDTDiscretizer、GBDTDiscretizer
+ 调用示例：
```
data = pd.read_csv(path, header=0)
y = data['Y']
X = data.drop(labels=['Y'], axis=1)
gbdis = GBDTDiscretizer(columns=X.columns, n_decimals=6)
sc = StandardScaler()
lr = LogisticRegression()
clf = Pipeline([('gbdis', gbdis), ('sc', sc), ('lr', lr)])
clf.fit(X_train, y_train)
# 查看列名
gbdis.get_column_names()
```

