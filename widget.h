#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include "widget2.h"

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

private slots:
    void on_pushButton_clicked();

    void on_ToSecondButton_clicked();

private:
    Ui::Widget *ui;
    widget2 *ui2;
};




#endif // WIDGET_H
