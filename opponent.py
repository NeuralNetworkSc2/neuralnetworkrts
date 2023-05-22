from sc2 import Race, Difficulty
from sc2.ids.unit_typeid import UnitTypeId
from util import strategy


class opponent:
    def __init__(self, bot):
        self.bot = bot
        self.known_race = None
        self.known_hq_location = None
        self.known_natural = None
        self.start_location = bot.enemy_start_locations
        self.next_potential_location = None
        self.army_strength = 0
        self.units = None
        self.structures = None
        self.strategies = set()
        self.too_close_distance = 0

        if bot.enemy_race != Race.Random:
            self._set_race(bot.enemy_race)

    def _set_enemy_hq_and_natural(self, pos):
        self.known_hq_location = pos
        locations = list(self.bot.expansion_locations)
        locations.remove(pos)
        self.known_natural = pos.closest(locations)

    def _set_race(self, race):
        self.known_race = race

    def one_base(self):
        if len(self.bot.enemy_start_locations) == 1:
            self._set_enemy_hq_and_natural(self.bot.enemy_start_locations[0])
            self.start_location = []
            self.too_close_distance = self.bot.start_location.distance_to(self.bot._game_info.map_center)

    def refresh(self):
        if self.bot.known_enemy_units:
            self.units = self.bot.known_enemy_units
            if self.known_race is None:
                self._set_race(self.units.first.race)
        else:
            self.units = None

        if self.bot.known_enemy_structures:
            self.structures = self.bot.known_enemy_structures
            self.check_rush()
            self.check_cannon_rush()
        else:
            self.structures = None
        self.check_proxy()

        if self.start_location:
            for i, base in enumerate(self.start_location):
                if self.bot.units.closest_distance_to(base) < 10:
                    self.start_location.pop(i)
                    if self.structures and self.structures.closest_distance_to(base) < 20:
                        if not self.known_hq_location:
                            self._set_enemy_hq_and_natural(base)

        if self.known_hq_location and self.bot.units.closest_distance_to(self.known_hq_location) < 3:
            if not self.structures or self.structures.closest_distance_to(self.known_hq_location) > 20:
                self.known_hq_location = None

    def is_close(self, distance=None):
        if not distance:
            distance = self.too_close_distance
        if self.bot.start_location.distance_to_closest(self.structures) < distance:
            return True
        return False

    def check_proxy(self):
        if self.structures:
            if self.is_close() and strategy.PROXY not in self.strategies:
                self.strategies.add(strategy.PROXY)
            elif not self.is_close() and strategy.PROXY in self.strategies:
                self.strategies.remove(strategy.PROXY)

    def check_rush(self):
        if self.bot.time < 60 * 5 and self.known_race == Race.Zerg and strategy.ZERGLING_RUSH not in self.strategies:
            opponent_pools = self.structures(UnitTypeId.SPAWNINGPOOL)
            if opponent_pools.exists:
                bot_pools = self.bot.units(UnitTypeId.SPAWNINGPOOL)
                if bot_pools.exists:
                    if opponent_pools.first.build_progress - bot_pools.first.build_progress > 0.2:
                        self.strategies.add(strategy.ZERGLING_RUSH)
                else:
                    self.strategies.add(strategy.ZERGLING_RUSH)

    def check_cannon_rush(self):
        if self.known_race == Race.Protoss:
            if strategy.CANNON_RUSH not in self.strategies and self.is_close(15):
                self.strategies.add(strategy.CANNON_RUSH)
            elif strategy.CANNON_RUSH in self.strategies and not self.is_close(15):
                self.strategies.remove(strategy.CANNON_RUSH)

    def get_next_scoutable_location(self, source_location=None):
        if source_location is None:
            source_location = self.bot.start_location
        if self.known_hq_location:
            return self.known_hq_location
        elif self.next_potential_location:
            return self.next_potential_location
        elif self.start_location:
            return self.start_location.closest_to(source_location).position
        else:
            return None

    def get_next_potential_building_closest_to(self, source):
        if self.structures:
            return self.structures.closest_to(source).position
        return self.get_next_scoutable_location(source)
