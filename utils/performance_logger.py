import time
import logging

# إعداد الـ Logging لملف bot_logs.txt
logging.basicConfig(
    filename="bot_logs.txt",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

class PerformanceLogger:
    def __init__(self):
        self.start_minerals = 50
        self.start_gas = 0
        self.start_time = None
        self.total_idle_time = 0
        self.last_step_time = None

    def setup_start_time(self, bot):
        """تشتغل مرة واحدة في on_start"""
        self.start_time = time.time()
        self.last_step_time = time.time()

    def log_performance(self, bot, iteration):
        """حساب معدلات الأداء وتسجيلها"""
        if self.start_time is None:
            self.setup_start_time(bot)

        current_time = time.time()
        elapsed_minutes = (current_time - self.start_time) / 60
        step_duration = current_time - self.last_step_time

        # 1. حساب معدل جمع الموارد في الدقيقة
        minerals_gained = bot.minerals - self.start_minerals
        gas_gained = bot.vespene - self.start_gas
        
        mpm = minerals_gained / elapsed_minutes if elapsed_minutes > 0 else 0
        gpm = gas_gained / elapsed_minutes if elapsed_minutes > 0 else 0

        # 2. تتبع العمال الـ Idle والكفاءة
        idle_workers = bot.workers.idle.amount
        if idle_workers > 0:
            self.total_idle_time += step_duration

        total_workers = bot.workers.amount
        efficiency = ((total_workers - idle_workers) / total_workers * 100) if total_workers > 0 else 0
        self.last_step_time = current_time

        # 3. طباعة البيانات في الـ Terminal
        print(f"\n--- STEP {iteration} PERFORMANCE ---")
        print(f"Minerals/Min: {mpm:.2f} | Gas/Min: {gpm:.2f}")
        print(f"Workers: {total_workers} (Idle: {idle_workers}) | Efficiency: {efficiency:.2f}%")
        print(f"Total Idle Time: {self.total_idle_time:.2f} sec")
        print("---------------------------------\n")

        # 4. تسجيل البيانات والـ Warnings لـ Terran في ملف الـ Logs
        logging.info(f"STEP={iteration} | MPM={mpm:.2f} | GPM={gpm:.2f} | IdleTime={self.total_idle_time:.2f} | Eff={efficiency:.2f}%")
        
        if idle_workers > 0:
            logging.warning(f"WARNING: {idle_workers} idle SCVs detected!")
            
        if bot.supply_left < 2:
            logging.warning("WARNING: Supply is almost blocked! Need Supply Depots.")