import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.bot_ai import BotAI

class MinimalBot(BotAI):
    async def on_step(self, iteration: int):
        if iteration == 0:
            print("Bot is alive!")

run_game(
    maps.get("Simple64"),
    [Bot(Race.Protoss, MinimalBot()), Computer(Race.Terran, Difficulty.VeryEasy)],
    realtime=False
)