# Bot

Это домашнее задание для DLS, суть которого в том, чтобы создать Telegram-бота, который переносит стиль с одной на другую
Видео-демонстрация работы бота: https://t.me/c/1822480932/2

## Установка (без докера)
### 1. Скопируте файлы
`git clone https://github.com/denis-gr/bot.git`
### 2. Сгенерируйте и добавте токен в переменное окружение
Токен можно получить в TG боте https://t.me/BotFather, просто перейдите в него и следуйте указаниям
Затем можно передать токен используя перенные окружения, например в cmd (для windows) `SET apikey=ВАШ_ТОКЕН`
### 3. Запустите
`python bot`
Обязательно передайте токен, если не сделали ранее
`python bot -t ВАШ_ТОКЕН`
Или
`python bot --token ВАШ_ТОКЕН`

## Установка (с помощью докера)
Для линукс
`sudo docker run --env "apikey=ВАШ_ТОКЕН" denisgrigoriev04/dsimagetgbot:0.1`

## Скриншоты
![image](https://user-images.githubusercontent.com/73753069/215632551-083cd0bb-77ab-4eb5-92b2-b6ac5dfe5926.png)
![image](https://user-images.githubusercontent.com/73753069/215632642-8471bf57-1752-493c-86c8-b2de4999a609.png)
