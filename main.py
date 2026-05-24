import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.bot_ai import BotAI

class MinimalBot(BotAI):
    async def on_step(self, iteration: int):
        if iteration == 0:
            print("Bot is alive as Terran!")

run_game(
    maps.get("Simple64"),
    # تم التعديل هنا لـ Race.Terran للبوت بتاعك
    [Bot(Race.Terran, MinimalBot()), Computer(Race.Zerg, Difficulty.VeryEasy)],
    realtime=False
)