from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId

from bot.economy_manager import EconomyManager
from utils.data_extractor import DataExtractor
from utils.performance_logger import PerformanceLogger
from utils.lstm_data_logger import LSTMDataLogger
from ai_models.feature_scaler import FeatureScaler
from ai_models.data_pipeline import get_final_ai_input

class TerranDataBot(BotAI):
    def __init__(self):
        super().__init__()
        # تجميع كل كلاسات التيم
        self.economy_manager = EconomyManager()
        self.extractor = DataExtractor()
        self.perf_logger = PerformanceLogger()
        self.data_logger = LSTMDataLogger(sequence_length=60)
        self.scaler = FeatureScaler()

    async def on_start(self):
        self.perf_logger.setup_start_time(self)

    async def on_step(self, iteration: int):
        # 1. تشغيل الاقتصاد (Terran)
        await self.economy_manager.manage_economy(self)

        # 2. محاكاة دفاع اللاعب البشري (لكي يطول الجيم ونجمع داتا)
        await self.simulate_human_player()

        # 3. دورة استخراج البيانات والـ Logging (كل 50 خطوة)
        if iteration % 50 == 0:
            self.perf_logger.log_performance(self, iteration)
            
            # سحب الداتا وعمل Scaling
            scaled_features = get_final_ai_input(self, self.extractor, self.scaler)
            
            # إرسالها لـ Logger أسامة لإضافة الـ Label (Starvation)
            self.data_logger.log_step(scaled_features, self)
            
            # حفظ الملف كل فترة لحماية الداتا
            if iteration % 500 == 0:
                self.data_logger.save_step_to_csv("training_data.csv")

    async def simulate_human_player(self):
        """دفاع حرس الحدود لحماية البوت أثناء جمع الداتا"""
        if not self.structures(UnitTypeId.COMMANDCENTER).ready.exists:
            return
        cc = self.structures(UnitTypeId.COMMANDCENTER).ready.first
        if self.can_afford(UnitTypeId.BARRACKS) and self.structures(UnitTypeId.BARRACKS).amount < 3:
            await self.build(UnitTypeId.BARRACKS, near=cc.position.towards(self.game_info.map_center, 8))
        for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.can_afford(UnitTypeId.MARINE):
                rax.train(UnitTypeId.MARINE)
        enemies_near_base = self.enemy_units.closer_than(20, cc.position)
        if enemies_near_base.exists:
            for marine in self.units(UnitTypeId.MARINE).idle:
                closest_enemy = enemies_near_base.closest_to(marine)
                marine.attack(closest_enemy)
        else:
            for marine in self.units(UnitTypeId.MARINE).idle:
                if marine.distance_to(cc.position) > 10:
                    marine.move(cc.position.towards(self.game_info.map_center, 5))