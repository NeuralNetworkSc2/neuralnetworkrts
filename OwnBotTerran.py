
# страта бота -- построить 4 казармы, производить головорезов и
# отправлять их по одному на все точки кристаллов пока не будет найден противник
# после нахождения, стакает армию и начинает спамить головорезами на чужой базе
# также юзает спелл гранату у головореза

# бот может слить только VeryHard сложности, шанс победы +- 20%

import random
import sc2
from sc2.player import Race, Difficulty
from sc2.position import Point2, Point3
from sc2.unit import Unit
from sc2.player import Computer, Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId


class OwnBot(sc2.BotAI):
    def __init__(self):
        self.combinedActions = []

    async def on_step(self, iteration):
        self.combinedActions = []
        if self.supply_left < 5 and self.townhalls.exists and self.supply_used >= 14 and self.can_afford(
                UnitTypeId.SUPPLYDEPOT) and self.units(UnitTypeId.SUPPLYDEPOT).not_ready.amount + self.already_pending(
                UnitTypeId.SUPPLYDEPOT) < 1:
            ws = self.workers.gathering
            if ws:
                w = ws.furthest_to(ws.center)
                loc = await self.find_placement(UnitTypeId.SUPPLYDEPOT, w.position, placement_step=3)
                if loc:
                    self.combinedActions.append(w.build(UnitTypeId.SUPPLYDEPOT, loc))

        if self.units(UnitTypeId.BARRACKS).ready.exists and self.can_afford(
                UnitTypeId.ORBITALCOMMAND):
            for cc in self.units(UnitTypeId.COMMANDCENTER).idle:
                self.combinedActions.append(cc(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND))

        for depot in self.units(UnitTypeId.SUPPLYDEPOT).ready:
            self.combinedActions.append(depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER))

        if 1 <= self.townhalls.amount < 2 and self.already_pending(UnitTypeId.COMMANDCENTER) == 0 and self.can_afford(
                UnitTypeId.COMMANDCENTER):
            next_expo = await self.get_next_expansion()
            location = await self.find_placement(UnitTypeId.COMMANDCENTER, next_expo, placement_step=1)
            if location:
                w = self.select_build_worker(location)
                if w and self.can_afford(UnitTypeId.COMMANDCENTER):
                    error = await self.do(w.build(UnitTypeId.COMMANDCENTER, location))
                    if error:
                        print(error)

        # создаем барраки и перед этим проверяем можем ли мы позволить это, если их < 4
        if self.units.of_type([UnitTypeId.SUPPLYDEPOT, UnitTypeId.SUPPLYDEPOTLOWERED,
                               UnitTypeId.SUPPLYDEPOTDROP]).ready.exists and self.units(
                UnitTypeId.BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS) < 4 and self.can_afford(
                UnitTypeId.BARRACKS):
            ws = self.workers.gathering
            if ws and self.townhalls.exists:
                w = ws.furthest_to(ws.center)
                loc = await self.find_placement(UnitTypeId.BARRACKS, self.townhalls.random.position, placement_step=4)
                if loc:
                    self.combinedActions.append(w.build(UnitTypeId.BARRACKS, loc))

        if self.can_afford(UnitTypeId.SCV) and self.supply_left > 0 and self.units(UnitTypeId.SCV).amount < 18 and (
                self.units(UnitTypeId.BARRACKS).ready.amount < 1 and self.units(
                UnitTypeId.COMMANDCENTER).idle.exists or self.units(UnitTypeId.ORBITALCOMMAND).idle.exists):
            for th in self.townhalls.idle:
                self.combinedActions.append(th.train(UnitTypeId.SCV))

        if self.units(UnitTypeId.BARRACKS).amount > 0 and self.already_pending(UnitTypeId.REFINERY) < 1:
            for th in self.townhalls:
                vgs = self.state.vespene_geyser.closer_than(10, th)
                for vg in vgs:
                    if await self.can_place(UnitTypeId.REFINERY, vg.position) and self.can_afford(UnitTypeId.REFINERY):
                        ws = self.workers.gathering
                        if ws.exists:
                            w = ws.closest_to(vg)
                            self.combinedActions.append(w.build(UnitTypeId.REFINERY, vg))

        # если хватает ресурсов и есть нужные постройки --  вызвать юнита Reaper
        if self.can_afford(UnitTypeId.REAPER) and self.supply_left > 0:
            for rax in self.units(UnitTypeId.BARRACKS).idle:
                self.combinedActions.append(rax.train(UnitTypeId.REAPER))

        # разбить рабочих по группам для добычи
        if iteration % 20 == 0:
            await self.distribute_workers()

        # контроль юнита Reaper
        for reaper in self.units(UnitTypeId.REAPER):
            #поиск вражеского наземного юнита в радиусе 5 игровых единиц
            enemyGroundUnits = self.known_enemy_units.not_flying.closer_than(5, reaper)
            if reaper.weapon_cooldown == 0 and enemyGroundUnits.exists:
                enemyGroundUnits = enemyGroundUnits.sorted(lambda x: x.distance_to(reaper))
                closestEnemy = enemyGroundUnits[0]
                self.combinedActions.append(reaper.attack(closestEnemy))
                continue
            # в случае нахождения врага рядом с головорезом и при этом сам головорез лоухп( < 35% ) то он должен отступить на конкретное расстояние
            enemyThreatsClose = self.known_enemy_units.filter(lambda x: x.can_attack_ground).closer_than(15, reaper)
            if reaper.health_percentage < 0.35 and enemyThreatsClose.exists:
                retreatPoints = self.veryCloseEnemy(reaper.position, distance=2) | self.veryCloseEnemy(reaper.position, distance=4)
                retreatPoints = {x for x in retreatPoints if self.inPathingGrid(x)}
                if retreatPoints:
                    closestEnemy = enemyThreatsClose.closest_to(reaper)
                    retreatPoint = closestEnemy.position.furthest(retreatPoints)
                    self.combinedActions.append(reaper.move(retreatPoint))
                    continue

            reaperGrenadeRange = self._game_data.abilities[AbilityId.KD8CHARGE_KD8CHARGE.value]._proto.cast_range
            enemyGroundUnitsInGrenadeRange = self.known_enemy_units.not_structure.not_flying.exclude_type(
                [UnitTypeId.LARVA, UnitTypeId.EGG]).closer_than(reaperGrenadeRange, reaper)
            if enemyGroundUnitsInGrenadeRange.exists and (reaper.is_attacking or reaper.is_moving):
                abilities = (await self.get_available_abilities(reaper))
                enemyGroundUnitsInGrenadeRange = enemyGroundUnitsInGrenadeRange.sorted(lambda x: x.distance_to(reaper),
                                                                                       reverse=True)
                furthestEnemy = None
                for enemy in enemyGroundUnitsInGrenadeRange:
                    if await self.can_cast(reaper, AbilityId.KD8CHARGE_KD8CHARGE, enemy, cached_abilities_of_unit=abilities):
                        furthestEnemy = enemy
                        break
                if furthestEnemy:
                    self.combinedActions.append(reaper(AbilityId.KD8CHARGE_KD8CHARGE, furthestEnemy))
                    continue

            enemyThreatsVeryClose = self.known_enemy_units.filter(lambda x: x.can_attack_ground).closer_than(4.5, reaper)
            if reaper.weapon_cooldown != 0 and enemyThreatsVeryClose.exists:
                retreatPoints = self.veryCloseEnemy(reaper.position, distance=2) | self.veryCloseEnemy(reaper.position, distance=4)
                retreatPoints = {x for x in retreatPoints if self.inPathingGrid(x)}
                if retreatPoints:
                    closestEnemy = enemyThreatsVeryClose.closest_to(reaper)
                    retreatPoint = max(retreatPoints, key=lambda x: x.distance_to(closestEnemy) - x.distance_to(reaper))
                    self.combinedActions.append(reaper.move(retreatPoint))
                    continue
            allEnemyGroundUnits = self.known_enemy_units.not_flying
            if allEnemyGroundUnits.exists:
                closestEnemy = allEnemyGroundUnits.closest_to(reaper)
                self.combinedActions.append(reaper.move(closestEnemy))
                continue

            # разведка 1 юнитом в рандом точку для поиска цели
            self.combinedActions.append(reaper.move(random.choice(self.enemy_start_locations)))

        for oc in self.units(UnitTypeId.ORBITALCOMMAND).filter(lambda x: x.energy >= 50):
            mfs = self.state.mineral_field.closer_than(10, oc)
            if mfs:
                mf = max(mfs, key=lambda x: x.mineral_contents)
                self.combinedActions.append(oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mf))

        if self.townhalls.exists:
            for w in self.workers.idle:
                th = self.townhalls.closest_to(w)
                mfs = self.state.mineral_field.closer_than(10, th)
                if mfs:
                    mf = mfs.closest_to(w)
                    self.combinedActions.append(w.gather(mf))



        await self.do_actions(self.combinedActions)

    #расчет позиций рядом с противником, в случае чего перемещения

    def veryCloseEnemy(self, position, distance=1):
        p = position
        d = distance
        return self.closeEnemy(position, distance) | {
            Point2((p.x - d, p.y - d)),
            Point2((p.x - d, p.y + d)),
            Point2((p.x + d, p.y - d)),
            Point2((p.x + d, p.y + d)),
        }

    def closeEnemy(self, position, distance=1):
        p = position
        d = distance
        return {
            Point2((p.x - d, p.y)),
            Point2((p.x + d, p.y)),
            Point2((p.x, p.y - d)),
            Point2((p.x, p.y + d)),
        }

    # передвижение наземных юнитов в точку
    def inPathingGrid(self, pos):
        # проверка на возможность перемещения(нужно пофиксить рядом с клифами и возвышенностями)
        assert isinstance(pos, (Point2, Point3, Unit))
        pos = pos.position.to2.rounded
        return self._game_info.pathing_grid[(pos)] != 0


    def already_pending(self, unit_type):
        ability = self._game_data.units[unit_type.value].creation_ability
        unitAttributes = self._game_data.units[unit_type.value].attributes

        buildings_in_construction = self.units.structure(unit_type).not_ready
        if 8 not in unitAttributes and any(o.ability == ability for w in (self.units.not_structure) for o in w.orders):
            return sum([o.ability == ability for w in (self.units - self.workers) for o in w.orders])
        elif any(o.ability == ability for w in self.workers for o in w.orders):
            return sum([o.ability == ability for w in self.workers for o in w.orders]) \
                - buildings_in_construction.amount
        elif any(o.ability.id == ability.id for w in (self.units.structure) for o in w.orders):
            return sum([o.ability.id == ability.id for w in (self.units.structure) for o in w.orders])
        elif any(egg.orders[0].ability == ability for egg in self.units(UnitTypeId.EGG)):
            return sum([egg.orders[0].ability == ability for egg in self.units(UnitTypeId.EGG)])
        return 0

    async def distribute_workers(self, performanceHeavy=True, onlySaturateGas=False):
        mineral = [x.tag for x in self.state.units.mineral_field]
        geyser = [x.tag for x in self.geysers]

        workerPool = self.units & []
        workerPoolTags = set()

        # поиск гейзеров
        deficitGeysers = {}
        surplusGeysers = {}
        for g in self.geysers.filter(lambda x: x.vespene_contents > 0):
            # работа только с теми, что имеют ресы
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

        # функция для добавления рабочих в случае нехватки ресурсов
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
    sc2.run_game(sc2.maps.get("AscensiontoAiurLE"), [
        Bot(Race.Terran, OwnBot()),
        Computer(Race.Zerg, Difficulty.VeryHard)
    ], realtime=False)

# Список протестированных карт
# AbyssalReefLE
# AscensiontoAiurLE
# BloodBoilLE
# DefendersLandingLe
# Odyssey LE
# (с приставкой LE есть какие-то траблы, ее то надо указывать отдельно через пробел, то вместе)
# пак карт Ladder2017Season2


# выписал все сложности ботов чтобы было проще
# VeryEasy пройден
# Easy пройден
# Medium пройден
# MediumHard пройден
# Hard пройден
# Harder пройден
# VeryHard пройден ( 1 из 5 примерно)
if __name__ == '__main__':
    main()
