import random
import sc2
from sc2 import Race, Difficulty
from sc2.position import Point2, Point3
from sc2.unit import Unit
from sc2.player import Computer, Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId

class ZergBot(sc2.BotAI):
    def __init__(self):
        super(ZergBot, self).__init__()
        self.combinedActions = []

    async def on_step(self, iteration):
        self.combinedActions = []

        self.distribute_workers()
        self.create_unit_by_th(UnitTypeId.DRONE, 17)
        self.create_unit_by_th(UnitTypeId.OVERLORD, 2)
        self.create_building(UnitTypeId.SPAWNINGPOOL)
        self.create_building(UnitTypeId.EXTRACTOR)
        self.create_unit_by_th(UnitTypeId.QUEEN, 2)
        self.create_unit_by_th(UnitTypeId.ZERGLING)
        self.create_unit_by_th(UnitTypeId.QUEEN, 2)
        if self.units(UnitTypeId.HATCHERY).amount == 2:
            self.create_unit_by(UnitTypeId.SPAWNINGPOOL,UnitTypeId.QUEEN, 2)
        pool = self.units(UnitTypeId.SPAWNINGPOOL).ready.idle
        if self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED):
            pool.first.research(UpgradeId.ZERGLINGMOVEMENTSPEED)
        self.expand_now()
        self.create_building(UnitTypeId.BANELINGNEST)


        await self.do_actions(self.combinedActions)


    async def create_unit_by_th(self, id: UnitTypeId, limit: int):
        for hatchery in self.units(UnitTypeId.HATCHERY).ready.idle:
            if self.can_afford(id) and self.units(id).count() <= limit:
                self.combinedActions.append(self.do(hatchery.train(id)))

    async def create_unit_by(self, building: UnitTypeId, id: UnitTypeId, limit: int):
        for hatchery in self.units(UnitTypeId.HATCHERY).ready.idle:
            if self.can_afford(id) and self.units(id).count() <= limit:
                self.combinedActions.append(self.do(hatchery.train(id)))


    async def create_building(self, id: UnitTypeId):
        if self.supply_left < 5 and not self.already_pending(id):
            hatch = self.units(UnitTypeId.HATCHERY).ready
            if hatch.exists:
                if self.can_afford(id) and self.units(id).count() < 2:
                    self.combinedActions.append(self.build(id, near=hatch.first))


if __name__ == "__main__":
    sc2.run_game(sc2.maps.get("AscensiontoAiurLE"), [
        Bot(Race.Terran, ZergBot()),
        Computer(Race.Zerg, Difficulty.VeryHard)
    ], realtime=False)


