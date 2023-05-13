#include "widget.h"
#include "ui_widget.h"
#include <QPixmap>

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui2 = new widget2;
    connect(ui2, &widget2::FirstWindow, this, &Widget::show);
    ui->setupUi(this);
}

Widget::~Widget()
{
    delete ui;
}


void Widget::on_pushButton_clicked() {
    system("python D:\\StarCraftAI\\ProtossBot.py");
}


void Widget::on_ToSecondButton_clicked() {
    this->close();
    ui2->show();
}

