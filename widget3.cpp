#include "widget3.h"
#include "ui_widget3.h"

widget3::widget3(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::widget3)
{
    ui->setupUi(this);
}

widget3::~widget3()
{
    delete ui;
}

void widget3::on_ToSecondButton_clicked() {
    this->close();
    emit SecondWindow();
}


void widget3::on_StartButton_clicked() {
    std::string dif[] = {"VeryEasy", "Easy", "Medium", "MediumHard", "Hard", "Harder", "VeryHard"};
    std::string rac[] = {"Protoss", "Terran", "Zerg"};

    QString Mapa = ui->MapBox->currentText();
    std::string Mapa2 = Mapa.toStdString();
    if (Mapa2 == "Random") {
        int p = rand() % maps.size();
        while(maps[p] == "Random") {
            p = rand() % maps.size();
        }
        Mapa2 = maps[p].toStdString();
        qDebug() << QString(Mapa2.c_str());
    }

    QString Difi = ui->BotDifficulty->currentText();
    std::string Difi2 = Difi.toStdString();
    if (Difi2 == "Random") {
        int p = rand() % 7;
        Difi2 = dif[p];
    }

    std::string RacaBot = ui->BotRace->currentText().toStdString();
    if (RacaBot == "Random") {
        int p = rand() % 3;
        RacaBot = rac[p];
    }

    std::string MyBot = ui->MyRace->currentText().toStdString();
    if (MyBot == "Random") {
        int p = rand() % 3;
        MyBot = rac[p];
    }

    std::string dir;
    if (MyBot == "Protoss") {
        dir = "D:\\StarCraftAI\\ProtossBot.py";
    }
    if (MyBot == "Terran") {
        dir = "D:\\StarCraftAI\\TerranBot.py";
    }
    qDebug() << QString(dir.c_str());
    std::string zapusk = "python " + dir + " " + Mapa2 + " " + Difi2 + " " + RacaBot;
    system(zapusk.c_str());
}

