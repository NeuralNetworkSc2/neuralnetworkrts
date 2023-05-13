#include "widget2.h"
#include "ui_widget2.h"
#include "ui_widget3.h"
#include <QFileDialog>
#include <filesystem>
#include <QMessageBox>
namespace fs = std::filesystem;

widget2::widget2(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::widget2)
{
    ui3 = new widget3;
    connect(ui3, &widget3::SecondWindow, this, &widget2::show);
    ui->setupUi(this);
}

widget2::~widget2()
{
    delete ui;
}

void widget2::on_ToFirstButton_clicked() {
    this->close();
    emit FirstWindow();
}


bool HaveMaps = false;

void widget2::on_ToThirdButton_clicked() {
    if (!HaveMaps) {
        QMessageBox::critical(this, tr("Обишка"), tr("В директории с картами нет карт"));
        return ;
    }
    ui3->ui->MapBox->clear();
    ui3->ui->MapBox->addItem(tr("Random"));
    for(auto& i : ui3->maps) {
        ui3->ui->MapBox->addItem(i);
    }
    this->close();
    ui3->show();
}


void widget2::on_ChoseDirectoryGame_clicked() {
    QString path = QFileDialog::getExistingDirectory();
    ui->lineEdit->setText(path);
}


void widget2::on_ChoseDirectoryMaps_clicked() {
    ui3->maps.clear();
    QString path = QFileDialog::getExistingDirectory();
    if (path.size() == 0) {
        return ;
    }
    ui->lineEdit_2->setText(path);
    std::string pth = path.toStdString();
    for (auto & p : fs::directory_iterator(pth)) {
        if ( p.path().extension() == ".SC2Map" ) {
            QString cur(p.path().string().c_str());
            bool f = false;
            QString rez;
            for(int i = cur.size() - 1; i >= 0; i--) {
                if (cur[i] == '\\' || cur[i] == '/') {
                    break;
                }
                if (f) {
                    rez.push_front(cur[i]);
                }
                if (cur[i] == '.') {
                    f = true;
                }
            }
            ui3->maps.append(rez);
        }
    }
    if (ui3->maps.size() == 0) {
        HaveMaps = false;
    } else {
        HaveMaps = true;
    }
}

