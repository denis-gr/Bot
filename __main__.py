# import
import os
import argparse
from collections import defaultdict
import logging

from aiogram import Bot, Dispatcher, executor, types

from model import process

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure arguments parser.
parser = argparse.ArgumentParser(description='DSImageTgBot')
parser.add_argument('--token', '-t', default=os.environ.get('apikey'), help='Api telegram token')

# Set settings
config = parser.parse_args()

# Initialize bot and dispatcher
bot = Bot(token="5800929219:AAEJkd1w3358zH5fkfiyqD0a0rIHrLicKus")
dp = Dispatcher(bot)

# Create temporary database
database = defaultdict(lambda: { "s": "", "c": "" })

# Handlers

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("""Привет, я DSImageTgBot!
Передай мне две картинки и я пенесу стиль с одной картинки (стиль-картинка) на другую (контент-картинка)
Первая попавшая в бот картинка - стиль-картинка, остальные контент-картинка
На каждую контент-картинка генеруется своя картинка со стиль-картинка
Можно указать явно какая картинка стиль-картинка или поменять её прислав картинку с подписью "Style" или "Стиль" (не зависимо от регистра), приэтом вам будет отправленна послдняя контент-картинка с этим стилем
""")

@dp.message_handler(content_types=["photo"])
async def get_image(mes: types.Message):
    if not database[mes.from_user.id]["s"]:
        database[mes.from_user.id]["s"] = mes.photo[-1].file_id
    elif mes.text and (mes.text.lower() in ["style", "стиль"]):
        database[mes.from_user.id]["s"] = mes.photo[-1].file_id
    else:
        database[mes.from_user.id]["c"] = mes.photo[-1].file_id
    
    if database[mes.from_user.id]["s"] and database[mes.from_user.id]["c"]:
        c_file_info = await bot.get_file(database[mes.from_user.id]["c"])
        s_file_info = await bot.get_file(database[mes.from_user.id]["s"])
        c_file = await bot.download_file(c_file_info.file_path)
        s_file = await bot.download_file(s_file_info.file_path)
        o_file = process(c_file, s_file)
        await mes.answer_photo(o_file)

# Running
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
