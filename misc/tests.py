import array
import unittest

import sc2.position
import numpy as np
import pysc2.lib.actions as ac
from map_list.list_of_maps import find as find_maps
from map_list.list_of_maps import isFounded
from test_runner import OwnBot
import torch
from torch.nn.functional import one_hot

def generate_zeroed_mask_dict():
    mask_dict = {}
    mask_dict["function"] = np.zeros((1, 1))
    mask_dict["time_skip"] = np.zeros((1, 1))

    for t in ac.TYPES:
        x = str(t)
        mask_dict[x] = np.zeros((1, 1))

    return mask_dict

def generate_zeroed_res_dict():
    result_dict = {}
    result_dict["function"] = np.zeros((1, 1), dtype=np.int32)
    result_dict["time_skip"] = np.zeros((1, 1), dtype=np.int32)

    for t in ac.TYPES:
        x = str(t)
        result_dict[x] = np.zeros((1, 1), dtype=np.int32)

    return result_dict


first_list = ['AbiogenesisLE.SC2Map', 'AcidPlantLE.SC2Map', 'BackwaterLE.SC2Map', 'BlackpinkLE.SC2Map',
                'CatalystLE.SC2Map', 'EastwatchLE.SC2Map', 'NeonVioletSquareLE.SC2Map']

mul_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mul_list2 = []
mul_list3 = [-1]
mul_list4 = [5643, 21534657, 7652243, 236457, 87346432, 34258697, 8673241, 125645]
mul_list5 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
             26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,
             51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75
            ,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]

path1 = 'C:\\Program Files (x86)\\StarCraft II'
path2 = 'C:\\Program Files (x86)'
path3 = 'C:\\'
path4 = 'D:\\'
path5 = 'D:\\pythonStarCraft'
path6 = 'C:\\Program Files (x86)\\Microsoft Visual Studio'

path7 = 'C:\\Program Files (x86)\\StarCraft II'
path8 = 'C:\\Program Files (x86)'
path9 = 'C:\\Program Files (x86)\\StarCraft II\\Maps\\Ladder2018Season1'
path10 = 'C:\\Program Files (x86)\\Common Files'
path11 = 'D:\\Pair'
class Tests(unittest.TestCase):

    def test_is_founded(self):
        self.assertEqual(isFounded('*.SC2Map', path1), True)
        self.assertEqual(isFounded('JagannathaAIE.SC2Map', path2), True)
        self.assertEqual(isFounded('*.SC2Map', path3), True)
        self.assertEqual(isFounded('not_real_map.SC2Map', path4), False)
        self.assertEqual(isFounded('*.SC2Map', path5), False)
        self.assertEqual(isFounded('*.SC2Map', path6), False)

    def test_what_founded(self):
        self.assertEqual(find_maps('BlackburnAIE.SC2Map', path7),['BlackburnAIE.SC2Map'])
        self.assertEqual(find_maps('Ladder2019Season3.zip', path8), ['Ladder2019Season3.zip'])
        self.assertEqual(find_maps('*.SC2Map', path9), first_list)
        self.assertEqual(find_maps('not_real_map.SC2Map', path10), [])
        self.assertEqual(find_maps('Pair.h', path11), ['Pair.h']) #тест для поиска любого другого файла, без привязки к карте

    def test_distance_to_enemy(self):
        point1 = sc2.position.Point2([15, 20])
        point2 = sc2.position.Point2([0, 0])
        point3 = sc2.position.Point2([1000, 1000])
        point4 = sc2.position.Point2([23667, 65435])
        point5 = sc2.position.Point2([-1, -1])
        point6 = sc2.position.Point2([1000000, 1000000])
        point7 = sc2.position.Point2([-0.000001, -0.000001])
        self.assertEqual(OwnBot.closeEnemy(OwnBot, point1, 10), {(5, 20), (15, 30), (25, 20), (15, 10)})
        self.assertEqual(OwnBot.closeEnemy(OwnBot, point2, 0), {(0, 0)})
        self.assertEqual(OwnBot.closeEnemy(OwnBot, point3, 0), {(1000, 1000)})
        self.assertEqual(OwnBot.closeEnemy(OwnBot, point4, 83524), {(-59857, 65435), (23667, 148959), (23667, -18089), (107191, 65435)})
        self.assertEqual(OwnBot.closeEnemy(OwnBot, point5, -1), {(-1, 0), (-1, -2), (0, -1), (-2, -1)})
        self.assertEqual(OwnBot.closeEnemy(OwnBot, point6, 500000), {(1000000, 500000), (500000, 1000000), (1000000, 1500000), (1500000, 1000000)})
        self.assertEqual(OwnBot.closeEnemy(OwnBot, point7), {(-1.000001, -1e-06),(-1e-06, -1.000001),(-1e-06, 0.999999),(0.999999, -1e-06)})
        self.assertEqual(OwnBot.closeEnemy(OwnBot, sc2.position.Point2([0, 10000000000]), 0), {(0, 10000000000)})

    def test_mul_all(self):
        self.assertEqual(mul_all(mul_list), 3628800)
        self.assertEqual(mul_all(mul_list2), 1)
        self.assertEqual(mul_all(mul_list3), -1)
        self.assertEqual(mul_all(mul_list4), 717020499209728083204843591666789925463204556161280)
        self.assertEqual(mul_all(mul_list5), 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000)

    def test_chunk_it(self):
        self.assertEqual(chunk_it([1, 2, 3, 4], 0), 'Error')
        self.assertEqual(chunk_it([], 0), 'Error')
        self.assertEqual(chunk_it([], 1000000), [])
        self.assertEqual(chunk_it([1, 2, 3, 4, 5], 10), [[], [1], [], [2], [], [3], [], [4], [], [5]])
        self.assertEqual(chunk_it([2], 5), [[], [], [], [], [2]])

    def test_generate_zeroed_dict(self):
        self.assertEqual(generate_zeroed_mask_dict(), dictionary)
        self.assertEqual(generate_zeroed_res_dict(), dictionary2)


if __name__ == '__main__':
    unittest.main()

dictionary = generate_zeroed_mask_dict()
dictionary2 = generate_zeroed_res_dict()
def chunk_it(seq, num):
    try: avg = len(seq) / float(num)
    except Exception as e:
        return 'Error'
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def mul_all(x):
    out = 1
    for v in x:
        out *= v
    return out

def screen_from_id(active, screen_pos):
    screen = one_hot(torch.zeros(1, 1).long(), num_classes=64 * 64)
    screen = screen.short()
    screen = screen * active
    screen = screen.reshape(screen_pos.shape + (1, 64, 64))
    return screen

def to_tensor(x, type=torch.float32):
    result = {}
    for field in x:
        if "feature" in field:
            result[field] = torch.tensor(x[field], dtype=None)
        else:
            result[field] = torch.tensor(x[field], dtype=type)
    return result


def concat_along_axis_tensor(x):
    output = {}
    for entry in x[0]:
        output[entry] = torch.cat([p[entry] for p in x])

    return output