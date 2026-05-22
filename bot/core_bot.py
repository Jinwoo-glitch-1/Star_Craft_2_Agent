from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId

class SBot(BotAI):
    async def on_step(self, iteration: int):
        # تقرير حالة البوت كل 100 إطار
        if iteration % 100 == 0:
            print(f"--- Frame {iteration} | Minerals: {self.minerals} | Workers: {self.workers.amount} | Supply: {self.supply_used}/{self.supply_cap} ---")
            
        # 1. التوزيع التلقائي للعمال (أفضل حل لإنهاء حالة الـ Idle)
        await self.distribute_workers()
        
        # 2. مهام البوت الأساسية
        await self.build_workers(iteration)
        await self.assign_workers_smartly()
        await self.build_supply()

    async def build_workers(self, iteration: int):
        worker_type = UnitTypeId.PROBE
        for nexus in self.townhalls.ready:
            # نتأكد إن الـ Nexus مش مشغول حالياً
            if len(nexus.orders) == 0:
                # حددنا هنا سقف 22 عامل للقاعدة لضمان التوازن الاقتصادي
                if self.workers.amount < self.townhalls.amount * 22:
                    if self.can_afford(worker_type) and self.supply_left > 0:
                        action = nexus.train(worker_type)
                        if action is not None:
                            await self.do(action)
                        print(f"🚀 Probe Trained! Supply left: {self.supply_left - 1}")

    async def assign_workers_smartly(self):
        """توزيع العمال للكريستالات التي تعاني من نقص"""
        for worker in self.workers.idle:
            if self.townhalls.ready:
                closest_nexus = self.townhalls.ready.closest_to(worker)
                local_minerals = self.state.mineral_field.closer_than(15, closest_nexus)

                if local_minerals:
                    deficient_minerals = local_minerals.filter(
                        lambda m: m.ideal_harvesters > m.assigned_harvesters
                    )
                    target_mineral = deficient_minerals.closest_to(worker) if deficient_minerals else local_minerals.closest_to(worker)
                    worker.gather(target_mineral)

    async def get_pylon_position(self, nexus):
        """تحديد مكان بناء استراتيجي وقابل للبناء"""
        for i in range(1, 10):
            pos = nexus.position.random_on_distance(i)
            if await self.can_place(UnitTypeId.PYLON, pos):
                return pos
        return None

    async def build_supply(self):
        """مراقبة الـ Supply وبناء Pylons في أماكن ذكية"""
        if self.supply_cap >= 200:
            return

        if self.supply_left < 4 and self.already_pending(UnitTypeId.PYLON) == 0:
            if self.can_afford(UnitTypeId.PYLON) and self.townhalls.ready:
                nexus = self.townhalls.ready.first
                pylon_pos = await self.get_pylon_position(nexus)
                
                if pylon_pos:
                    await self.build(UnitTypeId.PYLON, near=pylon_pos)
                    print(f"⚡ ACTION: Building Pylon at optimal position: {pylon_pos}")