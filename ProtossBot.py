import sc2
import random
from sc2.position import Point2, Point3
from sc2.unit import Unit
from sc2.player import Computer, Bot
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId
from sc2 import maps
from sc2.player import Bot, Computer
from sc2.main import run_game
from sc2.data import Race, Difficulty
from sc2.bot_ai import BotAI


class ProtossBot(sc2.BotAI):
    
    def __init__(self):
        self.ITERATIONS_PER_MINUTE = 1300
        self.MAX_WORKERS = 70
    
    async def on_step(self, iteraion):
        self.iteration = self.state.game_loop
        #print(self.state.game_loop)
        #расширяем территорию
        await self.expand()
        #распределяем рабочих
        await self.distribute_workers()
        #нанимаем рабочих
        await self.build_workers()
        #строим пилоны
        await self.build_pylons()
        #строим ассимиляторы
        await self.build_assimilators()
        #делаем ворота и кибернетик
        await self.build_army_buildings()
        #делаем сталкеров
        await self.build_army()
        #атакуем
        await self.attack()
    
    #наём рабочих
    async def build_workers(self):
        #смотрим чтобы в нексусе ничего было очереди и он может наняться, тогда нанимаем
        if (len(self.units(UnitTypeId.NEXUS)) * 18 > len(self.units(UnitTypeId.PROBE))):
            if (len(self.units(UnitTypeId.PROBE)) < self.MAX_WORKERS):
                for NEXUS in self.units(UnitTypeId.NEXUS).ready.noqueue: #HERE
                    if (self.can_afford(UnitTypeId.PROBE)):
                        await self.do(NEXUS.train(UnitTypeId.PROBE))
                        
    #строим пилоны
    async def build_pylons(self):
        #если осталось меньше 5 "еды" и мы можем себе позволить его то строим рядом с первым нексусом
        if (self.supply_left < 5 and not self.already_pending(UnitTypeId.PYLON)):
            nexuses = self.units(UnitTypeId.NEXUS).ready
            if (nexuses.exists):
                if (self.can_afford(UnitTypeId.PYLON)):
                    await(self.build(UnitTypeId.PYLON, near=nexuses.first))            
                
                

    #строим ассимиляторы
    async def build_assimilators(self):
        #проходим по всем нексусам и ассимилируем все источники в радиусе 25 клеток
        for NEXUS in self.units(UnitTypeId.NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(15.0, NEXUS)
            for vespene in vespenes:
                if (not self.can_afford(UnitTypeId.ASSIMILATOR)):
                    break
                worker = self.select_build_worker(vespene.position)
                if (worker is None):
                    break
                if (not self.units(UnitTypeId.ASSIMILATOR).closer_than(1.0, vespene).exists):
                    await self.do(worker.build(UnitTypeId.ASSIMILATOR, vespene))

    #расширение территории, строим больше нексусов
    async def expand(self):
        if (self.units(UnitTypeId.NEXUS).amount < 2 and self.can_afford(UnitTypeId.NEXUS) and not self.already_pending(UnitTypeId.NEXUS)):
             await self.expand_now()
    
    #строим звёздные врата и кибернетик
    async def build_army_buildings(self):
        if (self.units(UnitTypeId.PYLON).ready.exists):
            PYLON = self.units(UnitTypeId.PYLON).ready.random  
            if (self.units(UnitTypeId.GATEWAY).ready.exists and not self.units(UnitTypeId.CYBERNETICSCORE)):
                if (self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE)):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=PYLON)
            elif (len(self.units(UnitTypeId.GATEWAY)) < (self.iteration / self.ITERATIONS_PER_MINUTE / 4)):
                if (self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY)):
                    await self.build(UnitTypeId.GATEWAY, near=PYLON)        
            if (self.units(UnitTypeId.CYBERNETICSCORE).ready.exists):
                if (len(self.units(UnitTypeId.STARGATE)) < (self.iteration / self.ITERATIONS_PER_MINUTE / 4)):
                    if (self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE)):
                        await self.build(UnitTypeId.STARGATE, near=PYLON)

    #делаем сталкеров и пустотны[]
    async def build_army(self):
        for gw in self.units(UnitTypeId.GATEWAY).ready.noqueue: #HERE
            if (not self.units(UnitTypeId.STALKER).amount > self.units(UnitTypeId.VOIDRAY).amount or self.units(UnitTypeId.STALKER).amount < 5):
                if (self.can_afford(UnitTypeId.STALKER) and self.supply_left > 0):
                    await self.do(gw.train(UnitTypeId.STALKER)) 
        for sg in self.units(UnitTypeId.STARGATE).ready.noqueue: #HERE
            if (self.can_afford(UnitTypeId.VOIDRAY) and self.supply_left > 0):
                    await self.do(sg.train(UnitTypeId.VOIDRAY))

    #поиск цели для атаки
    def find_target(self, state):
        if (len(self.known_enemy_units) > 0):
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]
    
    #атака
    async def attack(self):
        
        #атака сталкера
        if (self.units(UnitTypeId.STALKER).amount > 10):
            for s in self.units(UnitTypeId.STALKER).idle:
                await self.do(s.attack(self.find_target(self.state)))  
        elif (self.units(UnitTypeId.STALKER).amount > 3):
            if (len(self.known_enemy_units) > 0):
                for s in self.units(UnitTypeId.STALKER).idle:
                    await self.do(s.attack(random.choice(self.known_enemy_units)))
                    
        #атака пустотного
        if (self.units(UnitTypeId.VOIDRAY).amount > 8):
            for s in self.units(UnitTypeId.VOIDRAY).idle:
                await self.do(s.attack(self.find_target(self.state)))  
        elif (self.units(UnitTypeId.VOIDRAY).amount > 2):
            if (len(self.known_enemy_units) > 0):
                for s in self.units(UnitTypeId.VOIDRAY).idle:
                    await self.do(s.attack(random.choice(self.known_enemy_units)))
                
    


run_game(
    maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, ProtossBot()),
    Computer(Race.Terran, Difficulty.Hard)
    ], realtime=False
)
     