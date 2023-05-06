import random
import sc2
from sc2.position import Point2, Point3
from sc2.unit import Unit
from sc2.player import Computer, Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty
from sc2.bot_ai import BotAI


class ZergBot(BotAI):
    def __init__(self):
        super(ZergBot, self).__init__()
        self.combinedActions = []

    async def on_step(self, iteration):
        self.combinedActions = []
        await self.distribute_workers()
        await self.create_unit_by_larva(UnitTypeId.OVERLORD, 2)
        await self.create_unit_by_larva(UnitTypeId.DRONE, 17)
        await self.build_geysir()
        # await self.create_building(UnitTypeId.SPAWNINGPOOL)
        # await self.create_building(UnitTypeId.EXTRACTOR)
        # await self.create_unit_by_larva(UnitTypeId.QUEEN, 2)
        # await self.create_unit_by_larva(UnitTypeId.ZERGLING, 6)
        # await self.create_unit_by_larva(UnitTypeId.QUEEN, 2)
        # if self.units(UnitTypeId.HATCHERY).amount == 2:
        #     await self.create_unit_by(UnitTypeId.SPAWNINGPOOL, UnitTypeId.QUEEN, 2)
        # pool = self.units(UnitTypeId.SPAWNINGPOOL).ready.idle
        # if self.can_afford(UpgradeId.ZERGLINGMOVEMENTSPEED):
        #     pool.first.research(UpgradeId.ZERGLINGMOVEMENTSPEED)
        # await self.expand_now()
        # await self.create_building(UnitTypeId.BANELINGNEST)


    async def build_geysir(self):
        for th in self.units(UnitTypeId.HATCHERY).ready:
            vaspenes = self.vespene_geyser.closer_than(1000.0, th)
            for vaspene in vaspenes:
                if not self.can_afford(UnitTypeId.EXTRACTOR):
                    break
                vaspene = self.vespene_geyser.first
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(UnitTypeId.EXTRACTOR).closer_than(1.0, vaspene).exists:
                    self.do(worker.build(UnitTypeId.EXTRACTOR, vaspene.position.to2))

    async def create_unit_by_larva(self, name: UnitTypeId, limit: int):
        for larva in self.larva.ready.idle:
            if self.can_afford(name) and len(self.units(name)) <= limit:
                print(len(self.units(name)))
                self.do(larva.train(name))

    async def create_unit_by(self, building: UnitTypeId, name: UnitTypeId, limit: int):
        for building in self.units(building).ready.idle:
            if self.can_afford(name) and len(self.units(name)) <= limit:
                self.do(building.train(name))

    async def create_building(self, name: UnitTypeId):
        for drone in self.units(UnitTypeId.DRONE).collecting:
            if self.can_afford(name) and len(self.units(name)) < 2:
                drone.build(name, self.units(UnitTypeId.GASCANISTERZERG).first.position)


if __name__ == "__main__":
    run_game(sc2.maps.get("AcropolisLE"), [
        Bot(Race.Zerg, ZergBot()),
        Computer(Race.Zerg, Difficulty.VeryHard)
    ], realtime=False)
    ZergBot()
