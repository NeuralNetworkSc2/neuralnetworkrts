from neuralnetworkrts.OwnBotTerran import OwnBot
import sys
import os
import openpyxl as opxl
import time
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
        map = str(current_sheet_test.cell_value(rowx=i,colx=0))
        list = map.split('.SC2Map')
        map = list[0]
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
        print(map)
        game_starts = time.time()
        result = sc2.run_game(sc2.maps.get(map), [
             Bot(Race.Terran, OwnBot()),
             Computer(sRace, sDifficulties[difficulty_index])
         ], realtime=False)
        game_ends = time.time() - game_starts
        print(game_ends, "aaaaaaaaaaa")
        first_sheet.write(i, 0, f"Test {i}")
        first_sheet.write(i, 1, str(result))
        first_sheet.write(i, 2, int(game_ends))
        result_book.save("result_of_tests.xls")
    first_sheet.write(current_sheet_test.nrows, 2, "Play-time / 5")
    result_book.save("result_of_tests.xls")
if __name__ == '__main__':
    main()
