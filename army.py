import random
import statistics
import time
from sc2.helpers import ControlGroup
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
import util
from util import strategy



class control_army:
    def __init__(self, bot):
        self.bot = bot
        self.opponent = bot.opponent

        self.first_overlord_tag = None
        self.first_overlord_ordered = False
        self.early_warning_overlord_tag = None
        self.early_warning_overlord_ordered = False

        self.has_verified_front_door = False
        self.all_combat_units = None
        self.reserve = ControlGroup([])
        self.harassing_base_scouts = ControlGroup([])
        self.no_mans_expansions_scouts = ControlGroup([])
        self.muta_flankers = ControlGroup([])
        self.base_defenders = ControlGroup([])

    def explore_first(self):
        self.first_overlord_tag = self.bot.units(UnitTypeId.OVERLORD).first.tag

    def refresh(self):
        self.all_combat_units = self.bot.units(UnitTypeId.ZERGLING).ready | self.bot.units(UnitTypeId.ROACH).ready | self.bot.units(UnitTypeId.HYDRALISK).ready | self.bot.units(UnitTypeId.MUTALISK).ready
        self.strength = util.get_units_strength(self.bot, self.all_combat_units)

        unassigned = self.all_combat_units.tags_not_in(self.reserve | self.harassing_base_scouts | self.no_mans_expansions_scouts | self.muta_flankers | self.base_defenders)
        if unassigned:
            self.reserve.add_units(unassigned)

        overlords = self.bot.units(UnitTypeId.OVERLORD)
        early_warning = overlords.find_by_tag(self.early_warning_overlord_tag)
        if not early_warning:
            volunteers = overlords.ready.tags_not_in([self.first_overlord_tag])
            if volunteers:
                self.early_warning_overlord_tag = volunteers.first.tag
                self.early_warning_overlord_ordered = False

        self._reinforce_from_reserve_if_empty(self.muta_flankers, UnitTypeId.MUTALISK, 10)
        self._reinforce_from_reserve_if_empty(self.harassing_base_scouts, UnitTypeId.ZERGLING, 1, True)
        if self.bot.time > 120:
            self._reinforce_from_reserve_if_empty(self.no_mans_expansions_scouts, UnitTypeId.ZERGLING, 1, True)

    def _reinforce_from_reserve_if_empty(self, group, unit_type, up_to=200, drone_fallback=False):
        survivors = group.select_units(self.bot.units)
        if not survivors:
            reserves = self.reserve.select_units(self.all_combat_units(unit_type)).take(up_to, require_all=False)
            for reserve in reserves:
                self.reserve.remove_unit(reserve)
                group.add_unit(reserve)
            if len(reserves) == 0 and drone_fallback:
                drones_available = self.bot.units(UnitTypeId.DRONE)
                if drones_available:
                    group.add_unit(drones_available.first)

    async def attack_with_one_unit(self):
        bot = self.bot
        if not bot.hq_loss_handled:
            actions = []
            bot.hq_loss_handled = True
            if bot.enemy_start_locations:
                for unit in bot.units(UnitTypeId.DRONE) | bot.units(UnitTypeId.QUEEN) | self.all_combat_units:
                    actions.append(unit.attack(bot.enemy_start_locations[0]))
                await bot.do_actions(actions)

    def find_nearest_quit(self):
        self.bot.ramps_distance_sorted = sorted(self.bot._game_info.map_ramps, key=lambda ramp: ramp.top_center.distance_to(self.bot.start_location))
        doors = []
        for ramp in self.bot.ramps_distance_sorted:
            if ramp.top_center.distance_to(self.bot.start_location) <= 30:
                doors.append(ramp)
        if len(doors) == 1:
            return doors[0].top_center
        else:
            return self.bot.start_location.towards(self.bot.game_info.map_center, 10)

    def _unit_dispersion(self, units):
        if units:
            center = units.center
            return statistics.median([unit.distance_to(center) for unit in units])
        else:
            return 0

    def get_seek_and_destroy_actions(self, units):
        actions = []
        for unit in units:
            if self.opponent.units:
                point = self.opponent.units.random.position.random_on_distance(random.randrange(5, 15))
            else:
                point = self.bot.map.get_random_point()
            actions.append(unit.attack(point))
        return actions

    def _large_enough_army(self, strength):
        enough = (150 + ((self.bot.time / 60) * 80))
        if strategy.PROXY in self.opponent.strategies:
            enough = 50
        return strength >= enough or self.bot.supply_used > 180

    def get_army_actions(self):
        actions = []
        units = self.reserve.select_units(self.bot.units)
        if units:
            next = None
            if self._large_enough_army(util.get_units_strength(self.bot, units)):
                next = self.bot.opponent.get_next_potential_building_closest_to(self.bot.army_attack_point)
                if next is None and strategy.HIDDEN_BASE not in self.opponent.strategies:
                    self.opponent.strategies.add(strategy.HIDDEN_BASE)
                elif next and strategy.HIDDEN_BASE in self.opponent.strategies:
                    self.opponent.strategies.remove(strategy.HIDDEN_BASE)

                if strategy.HIDDEN_BASE in self.opponent.strategies:
                    return self.get_seek_and_destroy_actions(units.idle)

                if next:
                    leader = units.closest_to(next)
                    if leader:
                        main_pack = units.closer_than(15, leader.position)
                        if main_pack.amount > 1:
                            dispersion = self._unit_dispersion(main_pack)
                            if dispersion >= 5:
                                next = leader.position
                        else:
                            next = units.center

            else:
                next = self.bot.hq_front_door
            self.bot.army_attack_point = next
            for unit in units:
                actions.append(unit.attack(self.bot.army_attack_point))

        return actions

    def air_destroy(self):
        actions = []
        mutas = self.muta_flankers.select_units(self.bot.units).idle
        if mutas:
            for muta in mutas:
                actions.append(muta.move(self.bot.map.flanker_waypoint, queue=False))
                actions.append(muta.move(self.bot.map.opponent_corner, queue=True))
                actions.append(muta.attack(self.opponent.known_hq_location, queue=True))
                actions.append(muta.attack(self.opponent.known_natural, queue=True))
        return actions

    def scout_and_attack(self):
        actions = []
        scouts = self.harassing_base_scouts.select_units(self.bot.units)
        if scouts:
            for scout in scouts:
                if self.opponent.known_hq_location and scout.distance_to(self.opponent.known_hq_location) < 3:
                    worker_enemies = self.opponent.units(UnitTypeId.DRONE) | self.opponent.units(UnitTypeId.PROBE) | self.opponent.units(UnitTypeId.SCV)
                    if worker_enemies and not scout.is_attacking:
                        victim = worker_enemies.closest_to(scout.position)
                        actions.append(scout.attack(victim))
                else:
                    location = self.opponent.get_next_scoutable_location()
                    if location:
                        actions.append(scout.move(location))
                if self.opponent.units:
                    enemies_closeby = self.opponent.units.filter(lambda unit: unit.can_attack_ground).closer_than(2, scout)
                    if enemies_closeby and scout.health_percentage < 0.4:
                        closest_enemy = enemies_closeby.closest_to(scout)
                        actions.append(scout.move(util.away(scout.position, closest_enemy.position, 4)))
                if not self.has_verified_front_door:
                    for ramp in self.bot._game_info.map_ramps:
                        if scout.distance_to(ramp.top_center) < 6:
                            self.has_verified_front_door = True
                            self.bot.hq_front_door = ramp.top_center
        return actions

    def scout_no_mans_expansions(self):
        actions = []
        scouts = self.no_mans_expansions_scouts.select_units(self.bot.units)
        if scouts.idle:
            exps = list(self.bot.expansion_locations)
            if self.opponent.known_hq_location:
                exps.remove(self.opponent.known_hq_location)
            if self.opponent.known_natural:
                exps.remove(self.opponent.known_natural)
            for scout in scouts:
                actions.append(scout.move(self.bot.hq_front_door, queue=False))
                for exp in exps:
                    actions.append(scout.move(exp, queue=True))
        return actions

    def patrol_with_overlords(self):
        actions = []
        overlords = self.bot.units(UnitTypeId.OVERLORD)
        firstborn = overlords.find_by_tag(self.first_overlord_tag)
        if firstborn and not self.first_overlord_ordered:
            if self.opponent.known_natural:
                near_enemy_front_door = self.opponent.known_natural.towards(self.opponent.known_hq_location, 4)
                safepoint_near_natural = util.away(self.opponent.known_natural, self.opponent.known_hq_location, 10)
                actions += [firstborn.move(near_enemy_front_door), firstborn.move(safepoint_near_natural, queue=True)]
            else:
                for enemy_loc in self.bot.enemy_start_locations:
                    actions.append(firstborn.move(enemy_loc, queue=True))
                actions.append(firstborn.move(self.bot.start_location, queue=True))
            self.first_overlord_ordered = True

        early_warner = overlords.find_by_tag(self.early_warning_overlord_tag)
        if early_warner:
            if strategy.PROXY not in self.opponent.strategies:
                if not self.early_warning_overlord_ordered:
                    hq = self.bot.start_location
                    center = self.bot.game_info.map_center
                    dist_between_hq_and_center = hq.distance_to(center)
                    halfway = hq.towards(center, dist_between_hq_and_center * 0.7)
                    actions.append(early_warner.move(halfway, queue=False))
                    actions.append(early_warner.patrol(halfway.random_on_distance(5), queue=True))
                    actions.append(early_warner.patrol(halfway.random_on_distance(5), queue=True))
                    self.early_warning_overlord_ordered = True
            else:
                actions.append(early_warner.move(self.bot.start_location, queue=False))

        if len(overlords) < 4:
            patrol = self.bot.hq_front_door.random_on_distance(random.randrange(3, 8))
        else:
            patrol = self.bot.start_location.random_on_distance(40)
        for overlord in overlords.idle.tags_not_in([self.first_overlord_tag, self.early_warning_overlord_tag]):
            actions.append(overlord.move(patrol))
        return actions

    def is_worker_rush(self, town, enemies_approaching):
        enemies = enemies_approaching.closer_than(6, town)
        worker_enemies = enemies(UnitTypeId.DRONE) | enemies(UnitTypeId.PROBE) | enemies(UnitTypeId.SCV)
        if worker_enemies.amount > 1 and (worker_enemies.amount / enemies.amount) >= 0.8:
            return True
        return False

    def future_destroyed(self, town):
        enemies = self.opponent.units.closer_than(6, town).exclude_type(UnitTypeId.OVERLORD)
        if enemies:
            return enemies
        else:
            if self.opponent.structures:
                buildings = self.opponent.structures.closer_than(15, town)
                if buildings:
                    return buildings
        return None

    def base_defend(self):
        actions = []
        for town in self.bot.townhalls:
            if self.opponent.units:
                enemies = self.future_destroyed(town)
                if enemies and enemies.not_flying:
                    enemy = enemies.closest_to(town)
                    new_defenders = self.reserve.select_units(self.all_combat_units).idle.closer_than(30, town)
                    self.reserve.remove_units(new_defenders)
                    self.base_defenders.add_units(new_defenders)
                    armed_and_existing_defenders = self.base_defenders.select_units(self.bot.units)
                    if not armed_and_existing_defenders:
                        drones = self.bot.units(UnitTypeId.DRONE).closer_than(15, town)
                        if drones:
                            self.base_defenders.add_units(drones)

                    all_defenders = self.base_defenders.select_units(self.bot.units)
                    if all_defenders:
                        for defender in all_defenders:
                            actions.append(defender.attack(enemy.position))

            if self.base_defenders and not (self.opponent.units and self.opponent.units.closer_than(10, town).exclude_type(UnitTypeId.OVERLORD)):
                defenders = self.base_defenders.select_units(self.bot.units)
                for unit in defenders:
                    self.base_defenders.remove_unit(unit)
                    if unit.type_id == UnitTypeId.DRONE:
                        actions.append(unit.move(town.position))
                    else:
                        self.reserve.add_unit(unit)
                        actions.append(unit.move(self.bot.hq_front_door))

        return actions

def can_research(bot, tech):
    if tech not in bot.state.upgrades and bot.can_afford(tech):
        if tech in [UpgradeId.ZERGGROUNDARMORSLEVEL2, UpgradeId.ZERGMISSILEWEAPONSLEVEL2, UpgradeId.ZERGFLYERWEAPONSLEVEL2, UpgradeId.ZERGFLYERARMORSLEVEL2]:
            if bot.units(UnitTypeId.LAIR).exists:
                return True
        elif tech in [UpgradeId.ZERGGROUNDARMORSLEVEL3, UpgradeId.ZERGMISSILEWEAPONSLEVEL3, UpgradeId.ZERGFLYERWEAPONSLEVEL3, UpgradeId.ZERGFLYERARMORSLEVEL3]:
            if bot.units(UnitTypeId.HIVE).exists:
                return True
        else:
            return True
    return False


def get_tech_to_research(bot, techs):
    for tech in techs:
        if can_research(bot, tech):
            return tech
    return None


def upgrade_tech(bot):
    if UpgradeId.GLIALRECONSTITUTION not in bot.state.upgrades and bot.can_afford(UpgradeId.GLIALRECONSTITUTION):
        if bot.units(UnitTypeId.ROACHWARREN).ready.exists and bot.units(UnitTypeId.LAIR).exists and bot.units(UnitTypeId.ROACHWARREN).ready.noqueue:
            return [bot.units(UnitTypeId.ROACHWARREN).ready.first.research(UpgradeId.GLIALRECONSTITUTION)]

    idle_chambers = bot.units(UnitTypeId.EVOLUTIONCHAMBER).ready.noqueue
    if idle_chambers:
        research_order = [
            UpgradeId.ZERGGROUNDARMORSLEVEL1,
            UpgradeId.ZERGGROUNDARMORSLEVEL2,
            UpgradeId.ZERGMISSILEWEAPONSLEVEL1,
            UpgradeId.ZERGMISSILEWEAPONSLEVEL2,
            UpgradeId.ZERGGROUNDARMORSLEVEL3,
            UpgradeId.ZERGMISSILEWEAPONSLEVEL3
        ]
        tech = get_tech_to_research(bot, research_order)
        if tech:
            return [idle_chambers.first.research(tech)]

    idle_spire = bot.units(UnitTypeId.SPIRE).ready.noqueue
    if idle_spire:
        research_order = [
            UpgradeId.ZERGFLYERWEAPONSLEVEL1,
            UpgradeId.ZERGFLYERWEAPONSLEVEL2,
            UpgradeId.ZERGFLYERARMORSLEVEL1,
            UpgradeId.ZERGFLYERARMORSLEVEL2,
            UpgradeId.ZERGFLYERWEAPONSLEVEL3,
            UpgradeId.ZERGFLYERARMORSLEVEL3
        ]
        tech = get_tech_to_research(bot, research_order)
        if tech:
            return [idle_spire.first.research(tech)]

    return []