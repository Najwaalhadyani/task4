# task4
1. تحميل البيانات واستعراضها
2. import pandas as pd
import numpy as np

# تحميل مجموعة بيانات تايتانيك
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# عرض أولى صفوف البيانات
df.head()
2. التعامل مع القيم المفقودة
# التعامل مع القيم المفقودة في العمود 'Age'
df['Age'].fillna(df['Age'].median(), inplace=True)

# التعامل مع القيم المفقودة في العمود 'Embarked'
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
3. تحويل القيم الفئوية إلى قيم عددية
# تحويل العمود 'Sex' من النصوص إلى قيم عددية
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# تحويل العمود 'Embarked' من النصوص إلى قيم عددية
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
4. تحديد ميزات النموذج والمتغير الهدف
# تحديد ميزات النموذج والمتغير الهدف
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']
5. تقسيم البيانات إلى مجموعات تدريب واختبار
from sklearn.model_selection import train_test_split

# تقسيم البيانات إلى مجموعات تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
6. التعامل مع القيم المفقودة في مجموعة التدريب
# التحقق من القيم المفقودة في مجموعة التدريب
print(X_train.isnull().sum())

# تعويض القيم المفقودة في مجموعة التدريب
X_train['Age'].fillna(X_train['Age'].median(), inplace=True)
X_train['Embarked'].fillna(X_train['Embarked'].mode()[0], inplace=True)

# إعادة التحقق من القيم المفقودة بعد التعويض
print(X_train.isnull().sum())
7. بناء وتدريب نموذج الانحدار اللوجستي
from sklearn.linear_model import LogisticRegression

# إنشاء نموذج الانحدار اللوجستي
model = LogisticRegression(max_iter=1000)

# تدريب النموذج على مجموعة التدريب
model.fit(X_train, y_train)
8. إجراء التنبؤات وتقييم النموذج
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# إجراء التنبؤات على مجموعة الاختبار
y_pred = model.predict(X_test)

# حساب دقة النموذج
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# عرض مصفوفة الالتباس
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# عرض تقرير التصنيف
print(classification_report(y_test, y_pred))
![task4](https://github.com/user-attachments/assets/8aec2a06-18c0-42f6-8fa2-bc26017da8d9)

<img width="417" alt="task 4_2" src="https://github.com/user-attachments/assets/5902b0ac-4bad-46c5-b9b4-f6103a7a19a1">

