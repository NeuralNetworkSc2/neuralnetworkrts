#ifndef WIDGET2_H
#define WIDGET2_H

#include <QWidget>
#include "widget3.h"

namespace Ui {
class widget2;
class widget3;
}

class widget2 : public QWidget
{
    Q_OBJECT

public:
    explicit widget2(QWidget *parent = nullptr);
    ~widget2();

signals:
    void FirstWindow();

private slots:
    void on_ToFirstButton_clicked();

    void on_ToThirdButton_clicked();

    void on_ChoseDirectoryGame_clicked();

    void on_ChoseDirectoryMaps_clicked();


private:
    Ui::widget2 *ui;
    widget3* ui3;
};

#endif // WIDGET2_H
