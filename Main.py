#%%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
import numpy as np

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.metrics import accuracy_score

from fancyimpute import KNN
from sklearn.metrics import confusion_matrix
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#%%
data = pd.read_csv('problem4_diabetes.csv', header=0)
#data = data.replace(r'\s+', np.nan, regex=True)
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

#%%
features.count()
(features == 0).astype(int).sum(axis=0)

features[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = features[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.nan)
(features == 0).astype(int).sum(axis=0)

#%%

features = pd.DataFrame(data=KNN(k=23).fit_transform(features), index=list(range(len(features))), columns=list(data.columns[:-1]))

#%%
plt.bar(x=0, height=(labels==0).sum(), width=0.5, color='salmon', label='Outcome 0')
plt.bar(x=1, height=(labels==1).sum(), width=0.5, color='cyan', label='Outcome 1')
plt.xticks([0, 1])
plt.xlabel('Outcome')
plt.ylabel('count')
plt.title('Distribution of Outcome variable')
plt.legend()
plt.show()

#%%

outcome0 = features['Pregnancies'].where(labels==0).dropna()
outcome1 = features['Pregnancies'].where(labels==1).dropna()

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(11,5))
fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

bp = ax1.boxplot([outcome0, outcome1],widths=0.7, notch=0, sym='+', vert=1, whis=1.5, labels=['Outcome0','Outcome1'])
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='o')

ax1.set_title('Number of Pregnancies Vs Diabetes')
ax1.set_xlabel('Outcome')
ax1.set_ylabel('Pregnancies')

boxCoords0 = np.column_stack([list(bp['boxes'][0].get_xdata()), list(bp['boxes'][0].get_ydata())])
boxCoords1 = np.column_stack([list(bp['boxes'][1].get_xdata()), list(bp['boxes'][1].get_ydata())])

ax1.add_patch(Polygon(boxCoords0, facecolor='salmon'))
ax1.add_patch(Polygon(boxCoords1, facecolor='c'))
ax1.plot(list(bp['medians'][0].get_xdata()), list(bp['medians'][0].get_ydata()), 'k')

ax1.set_ylim(0, 15)
ax1.set_xticklabels([0,1])

ax2.hist([outcome1, outcome0], color=['c', 'salmon'], label=['Outcome 0','Outcome 1'])
ax2.set_title('Pregnancies Vs Outcome')
ax2.set_xlabel('Pregnancies')
ax2.set_ylabel('count')

fig.text(0.18, 0.1,'Outcome 0', backgroundcolor='salmon', 
         color='black', weight='roman')
fig.text(0.28, 0.1, 'Outcome 1', backgroundcolor='c',
         color='black', weight='roman')
plt.legend()
plt.show()

#%%
for header in ['Glucose', 'BloodPressure', 'SkinThickness']:
    outcome0 = features[header].where(labels==0).dropna()
    outcome1 = features[header].where(labels==1).dropna()
    
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(11,5))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    
    bp = ax1.boxplot([outcome0, outcome1], widths=0.7,notch=0, sym='+', vert=1, whis=1.5, labels=['Outcome0','Outcome1'])
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='o')
    
    ax1.set_title('Variation of ' + header + ' Vs Diabetes')
    ax1.set_xlabel('Outcome')
    ax1.set_ylabel(header)
    
    boxCoords0 = np.column_stack([list(bp['boxes'][0].get_xdata()), list(bp['boxes'][0].get_ydata())])
    boxCoords1 = np.column_stack([list(bp['boxes'][1].get_xdata()), list(bp['boxes'][1].get_ydata())])
    
    ax1.add_patch(Polygon(boxCoords0, facecolor='salmon'))
    ax1.add_patch(Polygon(boxCoords1, facecolor='c'))
    ax1.plot(list(bp['medians'][0].get_xdata()), list(bp['medians'][0].get_ydata()), 'k')
    
    #ax1.set_ylim(50, 210)
    ax1.set_xticklabels([0,1])
    
    sns.distplot(outcome0, hist=False, kde=True, kde_kws={'shade':True, 'linewidth':3}, label='Outcome 0', color='salmon', ax=ax2)
    sns.distplot(outcome1, hist=False, kde=True, kde_kws={'shade':True, 'linewidth':3}, label='Outcome 1', color='c', ax=ax2)
    ax2.set_title('Density plot of ' + header)
    ax2.set_xlabel(header)
    ax2.set_ylabel('Density')
    #ax2.set_ylim(0, 0.02)
    
    fig.text(0.18, 0.1,'Outcome 0', backgroundcolor='salmon', 
             color='black', weight='roman')
    fig.text(0.28, 0.1, 'Outcome 1', backgroundcolor='c',
             color='black', weight='roman')
    plt.legend()
    plt.show()

#%%
idx = 0
for header in ['Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']:
    outcome0 = features[header].where(labels==0).dropna()
    outcome1 = features[header].where(labels==1).dropna()
    
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(11,5))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    
    bp = ax1.boxplot([outcome0, outcome1],widths=0.7, notch=0, sym='+', vert=1, whis=1.5, labels=['Outcome0','Outcome1'])
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='o')
    
    ax1.set_title('Variation of ' + header +' Vs Diabetes')
    ax1.set_xlabel('Outcome')
    ax1.set_ylabel(header)
    
    boxCoords0 = np.column_stack([list(bp['boxes'][0].get_xdata()), list(bp['boxes'][0].get_ydata())])
    boxCoords1 = np.column_stack([list(bp['boxes'][1].get_xdata()), list(bp['boxes'][1].get_ydata())])
    
    ax1.add_patch(Polygon(boxCoords0, facecolor='salmon'))
    ax1.add_patch(Polygon(boxCoords1, facecolor='c'))
    ax1.plot(list(bp['medians'][0].get_xdata()), list(bp['medians'][0].get_ydata()), 'k')
    
    ax1.set_xticklabels([0,1])
    
    bins = []
    bins.append(np.linspace(0, 750, 70))
    bins.append(np.linspace(0, 70, 30))
    bins.append(np.linspace(0, 2.5, 25))
    bins.append(np.linspace(0, 100, 20))
    ax2.hist(outcome0, bins[idx], alpha=0.5, color='salmon', label='Outcome 1')
    ax2.hist(outcome1, bins[idx], alpha=0.5, color='c', label='Outcome 0')
    idx = idx + 1
    ax2.set_title('Variation of ' + header +' Vs Diabetes')
    ax2.set_xlabel(header)
    ax2.set_ylabel('count')
    
    fig.text(0.18, 0.1,'Outcome 0', backgroundcolor='salmon', 
             color='black', weight='roman')
    fig.text(0.28, 0.1, 'Outcome 1', backgroundcolor='c',
             color='black', weight='roman')
    plt.legend()
    plt.show()
    
#%%
outcome0 = []
outcome1 = []  
for header in ['Age', 'Pregnancies', 'Insulin', 'Glucose', 'BMI', 'BloodPressure', 'SkinThickness']:
    outcome0.append(list(features[header].where(labels==0).dropna()))
    outcome1.append(list(features[header].where(labels==1).dropna()))

fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2, figsize=(11,10))
fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25, hspace=0.3)
    
ax1.scatter(outcome1[0], outcome1[1], color='c', label='1')
ax1.scatter(outcome0[0], outcome0[1], color='salmon', label='0')
ax1.set_title('Pregnancies & Age vs Diabetes')
ax1.set_xlabel('Age')
ax1.set_ylabel('Pregnancies')
ax1.legend(title='Outcome')

ax2.scatter(outcome1[2], outcome1[3], color='c', label='1')
ax2.scatter(outcome0[2], outcome0[3], color='salmon', label='0')
ax2.set_title('Insulin & Glucose vs Diabetes')
ax2.set_xlabel('Insulin')
ax2.set_ylabel('Glucose')
ax2.legend(title='Outcome')

ax3.scatter(outcome1[4], outcome1[5], color='c', label='1')
ax3.scatter(outcome0[4], outcome0[5], color='salmon', label='0')
ax3.set_title('BMI & BP vs Diabetes')
ax3.set_xlabel('BMI')
ax3.set_ylabel('BloodPressure')
ax3.legend(title='Outcome')

ax4.scatter(outcome1[4], outcome1[6], color='c', label='1')
ax4.scatter(outcome0[4], outcome0[6], color='salmon', label='0')
ax4.set_title('BMI & SkinThickness vs Diabetes')
ax4.set_xlabel('BMI')
ax4.set_ylabel('SkinThickness')
ax4.legend(title='Outcome')

plt.show()
#%%
corr = features.corr()
sns.heatmap(corr, annot=True, cmap='Blues',
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#%%
features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)
#%%
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=200)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
print(confusion_matrix(labels_test, pred))
print("Accuracy(Logistic Regression) = %.2f" % (accuracy*100) + "%")

#%%
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
print(confusion_matrix(labels_test, pred))
print("Accuracy(Random Forest) = %.2f" % (accuracy*100) , "%")

#%%
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
print(confusion_matrix(labels_test, pred))
print("Accuracy(Decision Tree) = %.2f" % (accuracy*100) , "%")


#%%
#graph = Source(tree.export_graphviz(clf , out_file=None, feature_names=list(data.columns.values), class_names=['0', '1'] , filled = True))
#display(SVG(graph.pipe(format='svg')))

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

tree.export_graphviz(clf, out_file='tree.dot', feature_names=features.columns.values, class_names=['Outcome0', 'Outcome1'], rounded=True, proportion=False, precision=2, filled=True)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in python
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree.png'))
plt.axis('off');
plt.show();