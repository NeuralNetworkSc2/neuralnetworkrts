import random
import sc2
from sc2 import Race, Difficulty
from sc2.position import Point2, Point3
from sc2.unit import Unit
from sc2.player import Computer, Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import  UpgradeId
from sc2.ids.ability_id import AbilityId


class protoss_bot(sc2.BotAI):
    def __init__(self):
        self.combinedActions = []

    async def on_step(self, iteration):
        self.combinedActions = []
        if self.can_afford(UnitTypeId.PROBE) and self.supply_left > 0 and self.units(UnitTypeId.PROBE).amount < 18 and (
                self.units(UnitTypeId.GATEWAY).ready.amount < 1 and self.units(UnitTypeId.NEXUS).idle.exists):
            for th in self.townhalls.idle:
                self.combinedActions.append(th.train(UnitTypeId.PROBE))

        if self.townhalls.exists:
            for w in self.workers.idle:
                th = self.townhalls.closest_to(w)
                mfs = self.state.mineral_field.closer_than(10, th)
                if mfs:
                    mf = mfs.closest_to(w)
                    self.combinedActions.append(w.gather(mf))

        if self.supply_left < 5 and not self.already_pending(UnitTypeId.PYLON):
            nexuses = self.units(UnitTypeId.NEXUS).ready
            if nexuses.exists:
                if self.can_afford(UnitTypeId.PYLON):
                    workers = self.workers.gathering
                    nex = self.townhalls.random
                    worker = workers.furthest_to(workers.center)
                    location = await self.find_placement(UnitTypeId.PYLON, nex.position, placement_step=3)
                    self.combinedActions.append(worker.build(UnitTypeId.PYLON, location))

        if self.units.of_type([UnitTypeId.PYLON]).ready.exists and self.units(
                UnitTypeId.GATEWAY).amount + self.already_pending(UnitTypeId.GATEWAY) + self.units(
                UnitTypeId.WARPGATE).amount + self.already_pending(UnitTypeId.WARPGATE) < 3 and self.can_afford(
                UnitTypeId.GATEWAY):
            ws = self.workers.gathering
            if ws and self.townhalls.exists:
                w = ws.furthest_to(ws.center)
                location = await self.find_placement(UnitTypeId.GATEWAY, self.townhalls.random.position, placement_step=4)
                if location:
                    self.combinedActions.append(w.build(UnitTypeId.GATEWAY, location))

        if self.units(UnitTypeId.GATEWAY).amount > 0 and self.already_pending(UnitTypeId.ASSIMILATOR) < 1:
            for th in self.townhalls:
                vespenes = self.state.vespene_geyser.closer_than(10, th)
                for vesp in vespenes:
                    if await self.can_place(UnitTypeId.REFINERY, vesp.position) and self.can_afford(UnitTypeId.ASSIMILATOR):
                        ws = self.workers.gathering
                        if ws.exists:
                            w = ws.closest_to(vesp)
                            self.combinedActions.append(w.build(UnitTypeId.ASSIMILATOR, vesp))
        #if self.units(UnitTypeId.ZEALOT).amount + self.already_pending(UnitTypeId.ZEALOT) < 5:
        #    if self.can_afford(UnitTypeId.ZEALOT) and self.supply_left > 0:
        #        for rax in self.units(UnitTypeId.GATEWAY).idle:
        #            self.combinedActions.append(rax.train(UnitTypeId.ZEALOT))

        if 1 <= self.townhalls.amount < 2 and self.already_pending(UnitTypeId.NEXUS) == 0 and self.can_afford(
                UnitTypeId.NEXUS):
            next_expo = await self.get_next_expansion()
            location = await self.find_placement(UnitTypeId.NEXUS, next_expo, placement_step=1)
            if location:
                w = self.select_build_worker(location)
                if w and self.can_afford(UnitTypeId.NEXUS):
                    error = await self.do(w.build(UnitTypeId.NEXUS, location))
                    if error:
                        print(error)

        if self.units.of_type([UnitTypeId.PYLON, UnitTypeId.GATEWAY]).ready.exists and self.units(
                UnitTypeId.FORGE).amount < 1 and self.units(
            UnitTypeId.GATEWAY).amount + self.already_pending(UnitTypeId.GATEWAY) > 0 and self.can_afford(
            UnitTypeId.FORGE):
            ws = self.workers.gathering
            if ws and self.townhalls.exists:
                w = ws.furthest_to(ws.center)
                location = await self.find_placement(UnitTypeId.FORGE, self.townhalls.random.position, placement_step=4)
                if location:
                    self.combinedActions.append(w.build(UnitTypeId.FORGE, location))

        if self.units(UnitTypeId.FORGE).amount > 0:
            forge = self.units(UnitTypeId.FORGE).first
            if self.can_afford(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1):
                self.combinedActions.append(forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1))
            if await self.on_upgrade_complete(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1) and self.can_afford(
                UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2):
                self.combinedActions.append(forge.research(UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2))
            if self.can_afford(UpgradeId.PROTOSSGROUNDARMORSLEVEL1):
                self.combinedActions.append(forge.research(UpgradeId.PROTOSSGROUNDARMORSLEVEL1))

        if self.units(UnitTypeId.CYBERNETICSCORE).amount > 0:
            csc = self.units(UnitTypeId.CYBERNETICSCORE).random
            if self.can_afford(UpgradeId.WARPGATERESEARCH):
                self.combinedActions.append(csc.research(UpgradeId.WARPGATERESEARCH))

        if self.units(UnitTypeId.CYBERNETICSCORE).amount == 0:
            if self.can_afford(UnitTypeId.CYBERNETICSCORE):
                workers = self.workers.gathering
                if workers and self.townhalls.exists:
                    solo_worker = workers.furthest_to(workers.center)
                    loca = await self.find_placement(UnitTypeId.CYBERNETICSCORE, self.townhalls.random.position, placement_step=3)
                    if loca:
                        self.combinedActions.append(solo_worker.build(UnitTypeId.CYBERNETICSCORE, loca))

        if self.units(UnitTypeId.CYBERNETICSCORE).amount == 1 and self.units(UnitTypeId.GATEWAY).amount > 0:
            if self.units(UnitTypeId.ROBOTICSFACILITY).amount == 0 and self.can_afford(UnitTypeId.ROBOTICSFACILITY):
                workers = self.workers.gathering
                if self.townhalls.exists and workers:
                    solo_worker = workers.furthest_to(workers.center)
                    location = await self.find_placement(UnitTypeId.ROBOTICSFACILITY, self.townhalls.first.position, placement_step=3)
                    if location:
                        self.combinedActions.append(solo_worker.build(UnitTypeId.ROBOTICSFACILITY, location))

        if self.units(UnitTypeId.GATEWAY).amount > 0 and self.units(UnitTypeId.CYBERNETICSCORE).amount > 0:
            if self.units(UnitTypeId.STALKER).amount + self.already_pending(UnitTypeId.STALKER) < 10 and self.supply_left > 2:
                if self.can_afford(UnitTypeId.STALKER) and self.supply_left > 0:
                    for gw in self.units(UnitTypeId.GATEWAY).idle:
                        self.combinedActions.append(gw.train(UnitTypeId.STALKER))

        if self.units(UnitTypeId.ROBOTICSFACILITY).amount > 0 and self.units(UnitTypeId.ROBOTICSBAY).amount \
                + self.already_pending(UnitTypeId.ROBOTICSBAY) < 1:
            workers = self.workers.gathering
            if workers and self.townhalls.exists:
                solo_worker = workers.furthest_to(workers.center)
                loc = await self.find_placement(UnitTypeId.ROBOTICSBAY, self.townhalls.random.position, placement_step=3)
                if loc:
                    self.combinedActions.append(solo_worker.build(UnitTypeId.ROBOTICSBAY, loc))



        if iteration % 25 == 0:
            await self.distribute_workers()

        await self.do_actions(self.combinedActions)



    async def distribute_workers(self, performanceHeavy=True, onlySaturateGas=False):
        mineral = [x.tag for x in self.state.units.mineral_field]
        geyser = [x.tag for x in self.geysers]

        workerPool = self.units & []
        workerPoolTags = set()

        # поиск гейзеров
        deficitGeysers = {}
        surplusGeysers = {}
        for g in self.geysers.filter(lambda x: x.vespene_contents > 0):
            deficit = g.ideal_harvesters - g.assigned_harvesters
            if deficit > 0:
                deficitGeysers[g.tag] = {"unit": g, "deficit": deficit}
            elif deficit < 0:
                surplusWorkers = self.workers.closer_than(10, g).filter(
                    lambda w: w not in workerPoolTags and len(w.orders) == 1 and w.orders[0].ability.id in [
                        AbilityId.HARVEST_GATHER] and w.orders[0].target in geyser)
                for i in range(-deficit):
                    if surplusWorkers.amount > 0:
                        w = surplusWorkers.pop()
                        workerPool.append(w)
                        workerPoolTags.add(w.tag)
                surplusGeysers[g.tag] = {"unit": g, "deficit": deficit}

        # просмотр всех точек по ресурсам
        deficitTownhalls = {}
        surplusTownhalls = {}
        if not onlySaturateGas:
            for th in self.townhalls:
                deficit = th.ideal_harvesters - th.assigned_harvesters
                if deficit > 0:
                    deficitTownhalls[th.tag] = {"unit": th, "deficit": deficit}
                elif deficit < 0:
                    surplusWorkers = self.workers.closer_than(10, th).filter(
                        lambda w: w.tag not in workerPoolTags and len(w.orders) == 1 and w.orders[0].ability.id in [
                            AbilityId.HARVEST_GATHER] and w.orders[0].target in mineral)
                    for i in range(-deficit):
                        if surplusWorkers.amount > 0:
                            w = surplusWorkers.pop()
                            workerPool.append(w)
                            workerPoolTags.add(w.tag)
                    surplusTownhalls[th.tag] = {"unit": th, "deficit": deficit}

            if all([len(deficitGeysers) == 0, len(surplusGeysers) == 0,
                    len(surplusTownhalls) == 0 or deficitTownhalls == 0]):
                return

        deficitGasCount = sum(
            gasInfo["deficit"] for gasTag, gasInfo in deficitGeysers.items() if gasInfo["deficit"] > 0)
        surplusCount = sum(-gasInfo["deficit"] for gasTag, gasInfo in surplusGeysers.items() if gasInfo["deficit"] < 0)
        surplusCount += sum(-thInfo["deficit"] for thTag, thInfo in surplusTownhalls.items() if thInfo["deficit"] < 0)

        if deficitGasCount - surplusCount > 0:
            for gTag, gInfo in deficitGeysers.items():
                if workerPool.amount >= deficitGasCount:
                    break
                workersNearGas = self.workers.closer_than(10, gInfo["unit"]).filter(
                    lambda w: w.tag not in workerPoolTags and len(w.orders) == 1 and w.orders[0].ability.id in [
                        AbilityId.HARVEST_GATHER] and w.orders[0].target in mineral)
                while workersNearGas.amount > 0 and workerPool.amount < deficitGasCount:
                    w = workersNearGas.pop()
                    workerPool.append(w)
                    workerPoolTags.add(w.tag)

        for gTag, gInfo in deficitGeysers.items():
            if performanceHeavy:
                workerPool.sort(key=lambda x: x.distance_to(gInfo["unit"]), reverse=True)
            for i in range(gInfo["deficit"]):
                if workerPool.amount > 0:
                    w = workerPool.pop()
                    if len(w.orders) == 1 and w.orders[0].ability.id in [AbilityId.HARVEST_RETURN]:
                        self.combinedActions.append(w.gather(gInfo["unit"], queue=True))
                    else:
                        self.combinedActions.append(w.gather(gInfo["unit"]))

        if not onlySaturateGas:
            for thTag, thInfo in deficitTownhalls.items():
                if performanceHeavy:
                    workerPool.sort(key=lambda x: x.distance_to(thInfo["unit"]), reverse=True)
                for i in range(thInfo["deficit"]):
                    if workerPool.amount > 0:
                        w = workerPool.pop()
                        mf = self.state.mineral_field.closer_than(10, thInfo["unit"]).closest_to(w)
                        if len(w.orders) == 1 and w.orders[0].ability.id in [AbilityId.HARVEST_RETURN]:
                            self.combinedActions.append(w.gather(mf, queue=True))
                        else:
                            self.combinedActions.append(w.gather(mf))

def main():
    sc2.run_game(sc2.maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, protoss_bot()),
        Computer(Race.Zerg, Difficulty.VeryEasy)
    ], realtime=False)

if __name__ == '__main__':
    main()
