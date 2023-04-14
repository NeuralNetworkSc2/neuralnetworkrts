import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, CYBERNETICSCORE, STALKER, STARGATE, VOIDRAY
import random


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
        if (len(self.units(NEXUS)) * 18 > len(self.units(PROBE))):
            if (len(self.units(PROBE)) < self.MAX_WORKERS):
                for nexus in self.units(NEXUS).ready.noqueue:
                    if (self.can_afford(PROBE)):
                        await self.do(nexus.train(PROBE))
                        
    #строим пилоны
    async def build_pylons(self):
        #если осталось меньше 5 "еды" и мы можем себе позволить его то строим рядом с первым нексусом
        if (self.supply_left < 5 and not self.already_pending(PYLON)):
            nexuses = self.units(NEXUS).ready
            if (nexuses.exists):
                if (self.can_afford(PYLON)):
                    await(self.build(PYLON, near=nexuses.first))            
                
                

    #строим ассимиляторы
    async def build_assimilators(self):
        #проходим по всем нексусам и ассимилируем все источники в радиусе 25 клеток
        for nexus in self.units(NEXUS).ready:
            vespenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vespene in vespenes:
                if (not self.can_afford(ASSIMILATOR)):
                    break
                worker = self.select_build_worker(vespene.position)
                if (worker is None):
                    break
                if (not self.units(ASSIMILATOR).closer_than(1.0, vespene).exists):
                    await self.do(worker.build(ASSIMILATOR, vespene))

    #расширение территории, строим больше нексусов
    async def expand(self):
        if (self.units(NEXUS).amount < 2 and self.can_afford(NEXUS) and not self.already_pending(NEXUS)):
             await self.expand_now()
    
    #строим звёздные врата и кибернетик
    async def build_army_buildings(self):
        if (self.units(PYLON).ready.exists):
            pylon = self.units(PYLON).ready.random  
            if (self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE)):
                if (self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE)):
                    await self.build(CYBERNETICSCORE, near=pylon)
            elif (len(self.units(GATEWAY)) < (self.iteration / self.ITERATIONS_PER_MINUTE / 4)):
                if (self.can_afford(GATEWAY) and not self.already_pending(GATEWAY)):
                    await self.build(GATEWAY, near=pylon)        
            if (self.units(CYBERNETICSCORE).ready.exists):
                if (len(self.units(STARGATE)) < (self.iteration / self.ITERATIONS_PER_MINUTE / 4)):
                    if (self.can_afford(STARGATE) and not self.already_pending(STARGATE)):
                        await self.build(STARGATE, near=pylon)

    #делаем сталкеров и пустотны[]
    async def build_army(self):
        for gw in self.units(GATEWAY).ready.noqueue:
            if (not self.units(STALKER).amount > self.units(VOIDRAY).amount or self.units(STALKER).amount < 5):
                if (self.can_afford(STALKER) and self.supply_left > 0):
                    await self.do(gw.train(STALKER)) 
        for sg in self.units(STARGATE).ready.noqueue:
            if (self.can_afford(VOIDRAY) and self.supply_left > 0):
                    await self.do(sg.train(VOIDRAY))

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
        if (self.units(STALKER).amount > 10):
            for s in self.units(STALKER).idle:
                await self.do(s.attack(self.find_target(self.state)))  
        elif (self.units(STALKER).amount > 3):
            if (len(self.known_enemy_units) > 0):
                for s in self.units(STALKER).idle:
                    await self.do(s.attack(random.choice(self.known_enemy_units)))
                    
        #атака пустотного
        if (self.units(VOIDRAY).amount > 8):
            for s in self.units(VOIDRAY).idle:
                await self.do(s.attack(self.find_target(self.state)))  
        elif (self.units(VOIDRAY).amount > 2):
            if (len(self.known_enemy_units) > 0):
                for s in self.units(VOIDRAY).idle:
                    await self.do(s.attack(random.choice(self.known_enemy_units)))
                
    


run_game(
    maps.get("AbyssalReefLE"), [
    Bot(Race.Protoss, ProtossBot()),
    Computer(Race.Terran, Difficulty.Hard)
    ], realtime=True
)