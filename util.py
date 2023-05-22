import random
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Pointlike
from sc2.position import Point2
from enum import Enum

def chance(percent):
    return random.randrange(100) < percent


def get_units_strength(bot, units):
    strength = 0
    for unit in units.filter(lambda u: u.can_attack_ground):
        if unit.type_id in [UnitTypeId.DRONE, UnitTypeId.SCV, UnitTypeId.PROBE]:
            strength += 10
        else:
            cost = bot._game_data.units[unit.type_id.value].cost
            strength += cost.minerals + cost.vespene
    return strength


def away(this: Pointlike, from_this: Pointlike, is_enough: float):
    distance_total = from_this.distance_to(this) + is_enough
    return from_this.towards(this, distance_total)

class map:
    def __init__(self, bot):
        self.bot = bot
        self.corners = []
        self.my_corner = None
        self.opponent_corner = None
        self.helper_corner = None
        self.flanker_waypoint = None
    def get_first_base(self):
        pos = self.bot.game_info.playable_area
        self.corners = [
            Point2((pos.x, pos.y)),
            Point2((pos.x + pos.width, pos.y)),
            Point2((pos.x, pos.y + pos.height)),
            Point2((pos.x + pos.width, pos.y + pos.height))
        ]
        enemy_base = self.bot.opponent.known_hq_location
        enemy_natural = self.bot.opponent.known_natural
        my_base = self.bot.start_location
        self.opponent_corner = enemy_base.closest(self.corners)
        self.my_corner = my_base.closest(self.corners)
        sorted_from_enemy_nat = enemy_natural.sort_by_distance(self.corners)
        self.helper_corner = sorted_from_enemy_nat[2]

        distance = self.helper_corner.distance_to(self.opponent_corner)
        self.flanker_waypoint = self.helper_corner.towards(self.opponent_corner, distance * 0.7)

    def get_random_point(self):
        pos = self.bot.game_info.playable_area
        x = random.randrange(pos.x, pos.x + pos.width)
        y = random.randrange(pos.y, pos.y + pos.height)
        return Point2((x, y))

class strategy(Enum):
    PROXY = 1
    ZERGLING_RUSH = 2
    HIDDEN_BASE = 3
    CANNON_RUSH = 4
