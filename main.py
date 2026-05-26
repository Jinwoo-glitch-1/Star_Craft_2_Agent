import os
import random
from sc2.data import Race, Difficulty
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.maps import get

# استدعاء البوت المجمع من core_bot
from bot.core_bot import TerranDataBot

if __name__ == "__main__":
    os.environ["SC2PATH"] = r"F:\StarCraft II"
    
    number_of_games = 15
    difficulties = [Difficulty.Medium, Difficulty.MediumHard, Difficulty.Hard]
    enemy_races = [Race.Zerg, Race.Protoss, Race.Terran]

    print(f"🚀 Starting Hardcore Data Collection for {number_of_games} games...")
    
    for game_num in range(number_of_games):
        current_diff = random.choice(difficulties)
        current_enemy = random.choice(enemy_races)
        
        print(f"\n====================================")
        print(f"⚔️ MATCH [{game_num + 1}/{number_of_games}] | Enemy: {current_enemy.name} | Difficulty: {current_diff.name}")
        print(f"====================================")
        
        bot_instance = TerranDataBot()
        
        try:
            run_game(
                get("NewRepugnancyLE"),
                [
                    Bot(Race.Terran, bot_instance),
                    Computer(current_enemy, current_diff)
                ],
                realtime=False
            )
            # حفظ الداتا النهائية للجيم بعد انتهائه
            bot_instance.data_logger.save_step_to_csv("training_data.csv")
        except Exception as e:
            print(f"⚠️ Game ended or crashed: {e}")
            continue
            
    print("\n✅ All games finished! 'training_data.csv' is ready for the LSTM model.")