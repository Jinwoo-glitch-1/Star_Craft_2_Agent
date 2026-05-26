from sc2.ids.unit_typeid import UnitTypeId

class EconomyManager:
    def __init__(self):
        pass

    async def manage_economy(self, bot):
        """تشغيل لوجيك العمال والـ Supply Depots للـ Terran"""
        
        # 1. تشغيل الـ Auto-Distribution (توزيع العمال التلقائي)
        await bot.distribute_workers()

        # 2. إنتاج عمال الـ Terran (SCVs) لحد حد معين (مثلاً 22)
        for cc in bot.townhalls.ready.idle:
            if bot.can_afford(UnitTypeId.SCV) and bot.workers.amount < 22:
                cc.train(UnitTypeId.SCV)

        # 3. بناء Supply Depot تلقائياً لو الـ Supply قرب يقفل
        if (
            bot.supply_left < 4
            and bot.already_pending(UnitTypeId.SUPPLYDEPOT) == 0
            and bot.can_afford(UnitTypeId.SUPPLYDEPOT)
        ):
            workers = bot.workers.gathering
            # التعديل هنا: شرط الأمان (لازم نتأكد إن عندنا Command Center لسة عايش وإن في عمال)
            if workers.exists and bot.townhalls.exists:
                worker = workers.random
                # البناء في اتجاه مركز الخريطة بقيمة 8 خطوات من الـ Command Center
                location = bot.townhalls.first.position.towards(bot.game_info.map_center, 8)
                await bot.build(UnitTypeId.SUPPLYDEPOT, near=location)