from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
import util

def get_drone_actions(self):
    return drones_to_minerals(self) + drones_to_extractors(self)

def drones_to_minerals(bot):
    actions = []
    for drone in bot.units(UnitTypeId.DRONE).idle:
        new_hatch = get_town_with_free_jobs(bot.townhalls)
        if new_hatch:
            mineral = get_closest_mineral_for_hatchery(bot.state.mineral_field(), new_hatch)
            actions.append(drone.gather(mineral))
    return actions

def drones_to_extractors(bot):
    actions = []
    for extractor in bot.units(UnitTypeId.EXTRACTOR):
        if extractor.assigned_harvesters < extractor.ideal_harvesters:
            worker = bot.workers.closer_than(20, extractor)
            if worker.exists:
                actions.append(worker.random.gather(extractor))
    return actions

def drone_rate_for_towns(townhalls):
    assigned_drones = 0
    ideal_drone_count = 0
    for town in townhalls:
        ideal_drone_count += town.ideal_harvesters
        assigned_drones += town.assigned_harvesters
    if ideal_drone_count == 0:
        return 1.0
    return assigned_drones / ideal_drone_count


def should_save_for_expansion(bot):
    if len(bot.townhalls.ready) > 1:
        if not bot.townhalls.not_ready and not bot.units.find_by_tag(bot.active_expansion_builder):
            if drone_rate_for_towns(bot.townhalls) >= 1.0 and len(bot.expansions_sorted) > 0:
                return True
    return False


def should_build_hatchery(bot):
    if not bot.townhalls.not_ready and not bot.units.find_by_tag(bot.active_expansion_builder):
        if drone_rate_for_towns(bot.townhalls) >= 0.8 and len(bot.expansions_sorted) > 0:
            if bot.minerals > 300 + (0 * (len(bot.townhalls) - 1)):
                return True
    return False


def get_town_with_free_jobs(townhalls, excluded=None):
    for town in townhalls:
        if town.assigned_harvesters < town.ideal_harvesters:
            if excluded is not None:
                if town != excluded:
                    return town
            else:
                return town
    return None


def get_expansion_order(expansion_locations, start_location):
    exps = expansion_locations
    exps.pop(start_location)
    sorted = start_location.sort_by_distance(exps)
    return sorted


async def reassign_drones(bot):
    for old_town in bot.townhalls:
        if old_town.assigned_harvesters > old_town.ideal_harvesters:
            drone = get_drone(old_town, bot.workers)
            new_hatch = get_town_with_free_jobs(bot.townhalls, old_town)
            if new_hatch and drone:
                mineral = get_closest_mineral_for_hatchery(bot.state.mineral_field(), new_hatch)
                await bot.do_actions([drone.gather(mineral)])


def get_drone(town, workers):
    workers = workers.closer_than(10, town)
    for worker in workers:
        if len(worker.orders) == 1 and worker.orders[0].ability.id in {AbilityId.HARVEST_GATHER, AbilityId.HARVEST_RETURN}:
            return worker
    if workers:
        return workers.random
    else:
        return None


def should_train_drone(bot, townhall):
    if len(bot.units(UnitTypeId.DRONE)) < 70:
        if townhall.assigned_harvesters < townhall.ideal_harvesters and bot.can_afford(UnitTypeId.DRONE):
            if len(bot.townhalls) == 1:
                chance = 100
            else:
                chance = 90
            return util.chance(chance)
    else:
        return False


def get_closest_mineral_for_hatchery(minerals, hatch):
    return minerals.closest_to(hatch.position)

async def produce_larvae(bot):
    actions = []
    for queen in bot.units(UnitTypeId.QUEEN).idle:
        abilities = await bot.get_available_abilities(queen)
        if AbilityId.EFFECT_INJECTLARVA in abilities:
            actions.append(queen(AbilityId.EFFECT_INJECTLARVA, bot.townhalls.closest_to(queen.position)))
    return actions

def set_hatchery(bot):
    actions = []
    for hatch in bot.townhalls:
        actions.append(hatch(AbilityId.RALLY_HATCHERY_UNITS, bot.hq_front_door))
        if not hatch.is_ready:
            actions.append(hatch(AbilityId.RALLY_HATCHERY_WORKERS, get_closest_mineral_for_hatchery(bot.state.mineral_field(), hatch)))
    return actions
