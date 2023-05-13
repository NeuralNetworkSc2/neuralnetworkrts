#ifndef WIDGET3_H
#define WIDGET3_H

#include <QWidget>

namespace Ui {
class widget3;
}

class widget3 : public QWidget
{
    Q_OBJECT

public:
    explicit widget3(QWidget *parent = nullptr);
    ~widget3();

signals:
    void SecondWindow();

private slots:
    void on_ToSecondButton_clicked();

    void on_StartButton_clicked();

public:
    QStringList maps;
    Ui::widget3 *ui;
};

#endif // WIDGET3_H
