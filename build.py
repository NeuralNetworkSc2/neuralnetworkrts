from sc2.ids.unit_typeid import UnitTypeId
import economy
from util import strategy
import util


class builder:
    def __init__(self, bot):
        self.bot = bot
        self.opponent = bot.opponent
        self.army = bot.army

    async def _build_one(self, it):
        if not (self.bot.units(it).exists or self.bot.already_pending(it)) and self.bot.can_afford(it):
            await self.bot.build(it, near=self.bot.townhalls.first.position.towards(self.bot._game_info.map_center, 5))

    async def _ensure_extractors(self):
        if self.bot.units(UnitTypeId.EXTRACTOR).ready.amount > 0 and not self.bot.units(UnitTypeId.LAIR).ready.exists:
            return
        elif not self.bot.already_pending(UnitTypeId.EXTRACTOR):
                for town in self.bot.townhalls:
                    if town.is_ready and economy.drone_rate_for_towns([town]) >= 0.90:
                        for geyser in self.bot.state.vespene_geyser.closer_than(10, town):
                            if await self.bot.can_place(UnitTypeId.EXTRACTOR, geyser.position) and self.bot.can_afford(UnitTypeId.EXTRACTOR):
                                workers = self.bot.workers.gathering
                                if workers.exists:
                                    worker = workers.closest_to(geyser)
                                    await self.bot.do_actions([worker.build(UnitTypeId.EXTRACTOR, geyser)])
                                    return

    def _should_train_overlord(self):
        if self.bot.can_afford(UnitTypeId.OVERLORD):
            if self.bot.units(UnitTypeId.OVERLORD).amount == 1:
                required_buffer = 0
            else:
                required_buffer = int((self.bot.townhalls.ready.amount + self.bot.units(UnitTypeId.QUEEN).ready.amount) * 0.7 + 2.5)
            buffer = self.bot.supply_left + (self.bot.already_pending(UnitTypeId.OVERLORD) * 8)
            should = buffer <= required_buffer and self.bot.supply_cap < 200
            return should

    async def create_buildings(self):
        random_townhall = self.bot.townhalls.first
        tech_penalty_multiplier = 1
        if {strategy.PROXY} & self.opponent.strategies:
            tech_penalty_multiplier = 2

        if economy.should_build_hatchery(self.bot):
            drone = self.bot.workers.random
            self.bot.active_expansion_builder = drone.tag
            await self.bot.do_actions([drone.build(UnitTypeId.HATCHERY, self.bot.expansions_sorted.pop(0))])
        if not economy.should_save_for_expansion(self.bot):
            await self._build_one(UnitTypeId.SPAWNINGPOOL)

            if self.bot.units(UnitTypeId.SPAWNINGPOOL).exists:
                await self._ensure_extractors()
            if self.bot.units(UnitTypeId.SPAWNINGPOOL).ready.exists:
                await self._build_one(UnitTypeId.ROACHWARREN)

            if self.bot.units(UnitTypeId.ROACHWARREN).ready.exists and self.army.strength >= 500 * tech_penalty_multiplier:
                if (not self.bot.units(UnitTypeId.LAIR).exists or self.bot.already_pending(UnitTypeId.LAIR)) and random_townhall.noqueue:
                    if self.bot.can_afford(UnitTypeId.LAIR):
                        await self.bot.do_actions([random_townhall.build(UnitTypeId.LAIR)])

                if self.bot.units(UnitTypeId.LAIR).ready.exists and len(self.bot.townhalls.ready) > 1 and self.army.strength >= 500 * tech_penalty_multiplier:
                    await self._build_one(UnitTypeId.EVOLUTIONCHAMBER)
                    await self._build_one(UnitTypeId.SPIRE)

    def train_units(self):
        actions = []
        for townhall in self.bot.townhalls:
            town_larvae = self.bot.units(UnitTypeId.LARVA).closer_than(5, townhall)
            if town_larvae.exists:
                larva = town_larvae.random
                if self._should_train_overlord():
                    actions.append(larva.train(UnitTypeId.OVERLORD))
                elif economy.should_train_drone(self.bot, townhall):
                    actions.append(larva.train(UnitTypeId.DRONE))
                elif not economy.should_save_for_expansion(self.bot):
                    if self.bot.can_afford(UnitTypeId.MUTALISK) and self.bot.units(UnitTypeId.SPIRE).ready.exists:
                        actions.append(larva.train(UnitTypeId.MUTALISK))
                    elif self.bot.units(UnitTypeId.ROACHWARREN).ready.exists:
                        if self.bot.can_afford(UnitTypeId.ROACH):
                            actions.append(larva.train(UnitTypeId.ROACH))
                        elif self.bot.minerals > 400 and self.bot.units(UnitTypeId.LARVA).amount > 5:
                            actions.append(larva.train(UnitTypeId.ZERGLING))
                    elif self.bot.can_afford(UnitTypeId.ZERGLING) and self.bot.units(UnitTypeId.SPAWNINGPOOL).ready.exists:
                        actions.append(larva.train(UnitTypeId.ZERGLING))
            if self.bot.units(UnitTypeId.SPAWNINGPOOL).ready.exists and townhall.is_ready and townhall.noqueue:
                if self.bot.can_afford(UnitTypeId.QUEEN):
                    if not self.bot.units(UnitTypeId.QUEEN).closer_than(15, townhall):
                        actions.append(townhall.train(UnitTypeId.QUEEN))
        return actions
