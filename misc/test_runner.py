from neuralnetworkrts.OwnBotTerran import OwnBot
import sys
import os
import openpyxl as opxl
import xlrd, xlwt
import sc2
from sc2 import Race, Difficulty
from sc2.player import Computer, Bot


test_workbook = xlrd.open_workbook_xls("tests_set.xls")
current_sheet_test = test_workbook["Sheet1"]
print(current_sheet_test.cell_value(rowx=0,colx=0))
difficulties = []
difficulties.append("VeryEasy")
difficulties.append("Easy")
difficulties.append("Medium")
difficulties.append("MediumHard")
difficulties.append("Hard")
difficulties.append("Harder")
difficulties.append("VeryHard")
sDifficulties = []
sDifficulties.append(Difficulty.VeryEasy)
sDifficulties.append(Difficulty.Easy)
sDifficulties.append(Difficulty.Medium)
sDifficulties.append(Difficulty.MediumHard)
sDifficulties.append(Difficulty.Hard)
sDifficulties.append(Difficulty.Harder)
sDifficulties.append(Difficulty.VeryHard)
result_book = xlwt.Workbook(encoding="utf-8")
first_sheet = result_book.add_sheet("Results")

def main():
    for i in range(current_sheet_test.nrows):
        sRace = 0
        sDifficulty = 0
        map = current_sheet_test.cell_value(rowx=i,colx=0)
        map = map.strip('.SC2Map')
        difficulty = current_sheet_test.cell_value(rowx=i,colx=1)
        race = current_sheet_test.cell_value(rowx=i,colx=2)
        print(map, difficulty, race)
        difficulty_index = difficulties.index(difficulty)
        print(difficulty_index)
        if race == 'Protoss':
            sRace = Race.Protoss
        elif race == 'Zerg':
            sRace = Race.Zerg
        elif race == 'Terran':
            sRace = Race.Terran
        result = sc2.run_game(sc2.maps.get(map), [
             Bot(Race.Terran, OwnBot()),
             Computer(sRace, sDifficulties[difficulty_index])
         ], realtime=False)
        first_sheet.write(i, 0, f"Test {i}")
        first_sheet.write(i, 1, str(result))
    result_book.save("result_of_tests.xls")
if __name__ == '__main__':
    main()
