from launch import *
from collection import *
from PyQt5 import QtGui
from PyQt5.QtWidgets import *

import sys
import os

class parentWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)

    def face_identification(self):
        os.system('./run.sh')

class childWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.child_ui = Ui_Dialog()
        self.child_ui.setupUi(self)
        self.label = QLabel(self)
        self.label.setFixedSize(200, 230)
        self.label.move(690, 680)

    def item_age(self):
        global age_temp
        age_temp = self.child_ui.Age.value()
        print(age_temp)

    def item_gender(self):
        global gender_temp
        gender_temp = self.child_ui.comboBox.currentIndex()
        print(gender_temp)

    def item_job(self):
        global job_temp
        job_temp = self.child_ui.comboBox_10.currentIndex()
        print(job_temp)

    def item_risk(self):
        global risk_temp
        risk_temp = self.child_ui.comboBox_7.currentIndex()
        print(risk_temp)

    def item_liabilities(self):
        global liabilities_temp
        liabilities_temp = self.child_ui.comboBox_2.currentIndex()
        print(liabilities_temp)

    def item_education(self):
        global education_temp
        education_temp = self.child_ui.comboBox_3.currentIndex()
        print(education_temp)

    def item_assets(self):
        global assets_temp
        assets_temp = self.child_ui.comboBox_4.currentIndex()
        print(assets_temp)

    def item_income(self):
        global income_temp
        income_temp = self.child_ui.comboBox_5.currentIndex()
        print(income_temp)

    def item_preference(self):
        global preference_temp
        preference_temp = self.child_ui.comboBox_8.currentIndex()
        print(preference_temp)

    def item_investment(self):
        global investment_temp
        investment_temp = self.child_ui.comboBox_9.currentIndex()
        print(investment_temp)

    def item_martial(self):
        global martial_temp
        martial_temp = self.child_ui.comboBox_6.currentIndex()
        print(martial_temp)

    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        jpg.save(
            "/home/bowen/catkin_ws/src/face_identification/configs/data/test_face_recognizer/images/compare_im/chenbowen1.jpg",
            "JPG", 100)

    def calculate(self):
        if age_temp < 22:
            score_age = 0
        else:
            score_age = 10

        if gender_temp == 0:
            score_gender = 10
        else:
            score_gender = 10

        if martial_temp == 0:
            score_martial = 5
        elif martial_temp == 1:
            score_martial = 10
        elif martial_temp == 2:
            score_martial = 8
        elif martial_temp == 3:
            score_martial = 6

        if job_temp == 0:
            score_job = 10
        elif job_temp == 1:
            score_job = 8
        elif job_temp == 2:
            score_job = 6
        elif job_temp == 3:
            score_job = 6
        elif job_temp == 4:
            score_job = 3
        elif job_temp == 5:
            score_job = 6
        elif job_temp == 6:
            score_job = 10
        elif job_temp == 7:
            score_job = 2

        if income_temp == 0:
            score_income = 2
        elif income_temp == 1:
            score_income = 4
        elif income_temp == 2:
            score_income = 6
        elif income_temp == 3:
            score_income = 8
        elif income_temp == 4:
            score_income = 9
        elif income_temp == 5:
            score_income = 10

        if preference_temp == 0:
            score_preference = 4
        elif preference_temp == 1:
            score_preference = 7
        elif preference_temp == 2:
            score_preference = 10

        if education_temp == 0:
            score_education = 4
        elif education_temp == 1:
            score_education = 6
        elif education_temp == 2:
            score_education = 8
        elif education_temp == 3:
            score_education = 10
        elif education_temp == 4:
            score_education = 2

        if liabilities_temp == 0:
            score_liabilities = 10
        elif liabilities_temp == 1:
            score_liabilities = 9
        elif liabilities_temp == 2:
            score_liabilities = 8
        elif liabilities_temp == 3:
            score_liabilities = 6
        elif liabilities_temp == 4:
            score_liabilities = 4

        if risk_temp == 0:
            score_risk = 10
        elif risk_temp == 1:
            score_risk = 0
        elif risk_temp == 2:
            score_risk = 5

        if assets_temp == 0:
            score_assets = 3
        elif assets_temp == 1:
            score_assets = 5
        elif assets_temp == 2:
            score_assets = 6
        elif assets_temp == 3:
            score_assets = 7
        elif assets_temp == 4:
            score_assets = 9
        elif assets_temp == 5:
            score_assets = 10

        if investment_temp == 0:
            score_investment = 2
        elif investment_temp == 1:
            score_investment = 4
        elif investment_temp == 2:
            score_investment = 8
        elif investment_temp == 3:
            score_investment = 10

        score_total = score_age + score_gender + score_martial + score_job + score_income + score_preference + score_education + score_liabilities + score_risk + score_assets + score_investment
        print(score_total)

        if score_total >= 85:
            print("The First Class")
        elif score_total >= 75 and score_total < 85:
            print("The Second Class")
        elif score_total >= 60 and score_total < 75:
            print("The Third Class")
        elif score_total >= 45 and score_total < 60:
            print("The Fourth Class")
        elif score_total < 45:
            print("The Fifth Class")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # my=pictureWindow()
    # my.show()

    font = QtGui.QFont()
    font.setPointSize(30)
    font.setBold(True)
    font.setWeight(75)

    age_temp = 18
    gender_temp = 0
    job_temp = 0
    risk_temp = 0
    liabilities_temp = 0
    education_temp = 0
    assets_temp = 0
    income_temp = 0
    preference_temp = 0
    investment_temp = 0
    martial_temp = 0

    window = parentWindow()
    child = childWindow()

    # 通过toolButton将两个窗体关联
    jumptoface = window.main_ui.pushButton
    jumptoface.clicked.connect(window.face_identification)
    jumptocollect = window.main_ui.pushButton_2
    jumptocollect.clicked.connect(child.show)

    age = child.child_ui.Age
    age.valueChanged.connect(child.item_age)

    gender = child.child_ui.comboBox
    gender.currentIndexChanged.connect(child.item_gender)

    liabilities = child.child_ui.comboBox_2
    liabilities.currentIndexChanged.connect(child.item_liabilities)

    education = child.child_ui.comboBox_3
    education.currentIndexChanged.connect(child.item_education)

    assets = child.child_ui.comboBox_4
    assets.currentIndexChanged.connect(child.item_assets)

    income = child.child_ui.comboBox_5
    income.currentIndexChanged.connect(child.item_income)

    martial = child.child_ui.comboBox_6
    martial.currentIndexChanged.connect(child.item_martial)

    risk = child.child_ui.comboBox_7
    risk.currentIndexChanged.connect(child.item_risk)

    preference = child.child_ui.comboBox_8
    preference.currentIndexChanged.connect(child.item_preference)

    investment = child.child_ui.comboBox_9
    investment.currentIndexChanged.connect(child.item_investment)

    job = child.child_ui.comboBox_10
    job.currentIndexChanged.connect(child.item_job)

    jumptoselect = child.child_ui.pushButton
    jumptoselect.clicked.connect(child.openimage)

    jumptocalculate = child.child_ui.pushButton_2
    jumptocalculate.clicked.connect(child.calculate)

    # 显示
    window.show()
    sys.exit(app.exec_())