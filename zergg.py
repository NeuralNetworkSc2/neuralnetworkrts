import random
import sys
import time
import sc2
from sc2 import Difficulty, Race
from sc2.data import race_townhalls
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot, Computer
import economy, army
from army import control_army
from opponent import opponent
from build import builder
from util import map

class Timer:
    def __init__(self, bot, ring_every=1.0):
        self.bot = bot
        self.last_ring = 0.0
        self.ring_every = ring_every

    def rings(self):
        if (self.bot.time - self.last_ring) >= self.ring_every:
            self.last_ring = self.bot.time
            return True
        else:
            return False


class zergg(sc2.BotAI):
    def on_start(self):
        self.opponent = opponent(self)
        self.army = control_army(self)
        self.builder = builder(self)
        self.drone_eco_optimization_timer = Timer(self, 0.2)
        self.map = map(self)
        self.army_timer = Timer(self, 0.05)
        self.build_timer = Timer(self, 0.5)
        self.active_expansion_builder = None
        self.expansions_sorted = []
        self.ramps_distance_sorted = None
        self.first_step = True
        self.hq_loss_handled = False
        self.hq_front_door = None
        self.army_attack_point = None

    def on_first_step(self):
        self.first_step = False
        start = time.time()
        self.expansions_sorted = economy.get_expansion_order(self.expansion_locations, self.start_location)
        self.hq_front_door = self.army.find_nearest_quit()
        self.army_attack_point = self.hq_front_door
        self.opponent.one_base()
        self.army.explore_first()
        self.map.get_first_base()

    async def on_step(self, iteration):
        step_start = time.time()
        budget = self.time_budget_available
        if budget and budget >= 0.3:
            await self.main_loop()

    async def main_loop(self):
        if self.first_step:
            self.on_first_step()
            return
        else:
            self.opponent.refresh()
            self.army.refresh()
        if not self.townhalls.exists:
            await self.army.attack_with_one_unit()
            return

        actions = []

        if self.drone_eco_optimization_timer.rings:
            await economy.reassign_drones(self)
            actions += economy.get_drone_actions(self)

        if self.army_timer.rings:
            actions += self.army.get_army_actions()
            actions += self.army.patrol_with_overlords()
            actions += self.army.scout_and_attack()
            actions += self.army.scout_no_mans_expansions()
            actions += self.army.air_destroy()
            actions += self.army.base_defend()

        if self.build_timer.rings:
            actions += economy.set_hatchery(self)
            actions += self.builder.train_units()
            await self.builder.create_buildings()
            actions += army.upgrade_tech(self)
            actions += await economy.produce_larvae(self)

        await self.do_actions(actions)
