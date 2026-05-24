import sc2
import random
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.bot_ai import BotAI

# ==========================================
# البوت بتاعك (Terran Macro/Turtle Bot)
# ==========================================
class TerranMasterBot(BotAI):
    async def on_step(self, iteration: int):
        # هنا هتحط كود البناء بتاع محمود وكود الـ Logger بتاعك
        if iteration == 0:
            print(f"Game Started! Playing against: {self.enemy_race.name}")

# ==========================================
# دالة تجميع البيانات (Offline Training Pipeline)
# ==========================================
def collect_training_data(num_games=10):
    """
    بيشغل اللعبة عدد معين من المرات لتجميع الداتا
    وبيبدل الصعوبة والخصم عشوائياً في كل جيم.
    """
    
    # 1. قائمة بالصعوبات اللي عايزين الموديل يتدرب عليها بالتدريج
    difficulties = [
        Difficulty.Medium, 
        Difficulty.MediumHard, 
        Difficulty.Hard
    ]
    
    # 2. قائمة بأعراق الخصوم عشان الموديل يشوف كل أنواع الهجوم
    enemy_races = [Race.Zerg, Race.Protoss, Race.Terran]

    print(f"🚀 Starting Data Collection for {num_games} games...")

    for i in range(num_games):
        # اختيار عشوائي للخصم والصعوبة
        current_enemy = random.choice(enemy_races)
        current_difficulty = random.choice(difficulties)
        
        print(f"\n==============================================")
        print(f"⚔️ Match [{i+1}/{num_games}] | Enemy: {current_enemy.name} | Difficulty: {current_difficulty.name}")
        print(f"==============================================")

        try:
            # تشغيل الجيم
            run_game(
                maps.get("Simple64"),
                [
                    Bot(Race.Terran, TerranMasterBot()),
                    Computer(current_enemy, current_difficulty)
                ],
                realtime=False  # مهم جداً: False عشان الجيم يخلص في ثواني/دقائق بدل وقت اللعب الحقيقي
            )
        except Exception as e:
            # حماية للكود: لو جيم هنج أو قفل فجأة، الـ Loop ماتقفش وتكمل الجيم اللي بعده
            print(f"⚠️ Game {i+1} crashed or ended unexpectedly: {e}")
            continue
            
    print("\n✅ All games finished successfully! Check your CSV files.")

# ==========================================
# نقطة التشغيل
# ==========================================
if __name__ == "__main__":
    # تقدر تخليها 100 قبل ما تنام وتسيب الجهاز يجمع داتا براحته
    collect_training_data(num_games=5)