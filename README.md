# VK-click-stream-AI
### Постановка задачи
- При помощи методов искусственного интеллекта предстоит предсказать, будет ли совершено определенное действие пользователем или нет в зависимости от его кликстрима   
- Задача построена на неструктурированных текстовых источниках данных   
- Каждый объект выборки характеризуется набором интернет-сессий (непрерывные последовательности посещенных пользователем сайтов)   
- Решение позволит сформировать спектр услуг под приоритеты и потребности отдельного пользователя   
- Решение задачи осложняется низкой долей положительного класса, наличием пропусков и зашумленности в данных, необходимостью выделения признаков из не структурированного текстового источника   
### Подход к решению
1. Вывели несколько моделей:
    - Кросс-валидация на выборке топ самых популярных слов - по предварительным подсчётам хорошо себя показал random forest
    - Вектор из "набора ключевых фраз" сравнили с "вектором из тренировочной выборки" ака использовали NLP библиотеку
    - Сравнили каждый вектор из теста с каждым вектором из трейна, усреднили (IN PROGRESS FOREVER)

2. Технические особенности:

    - Word2Vec, gensim, pymystem3, база от ruscorpora, scipy, Python, Numpy, Pandas, Matplotlib, VK Cloud, Jupiter, Colab,

3. Переход к анализу "векторов из ключевых фраз" и грамотный подход к применению NLP в анализе тайтлов

### bouns Интерпретируемость данных bonus

#### Ключевые слова, больше всего положительно влияющие на действие пользователя
![image](https://user-images.githubusercontent.com/75137969/187067955-55b3049b-5ff3-4f3e-8d77-e4930beb69e6.png)

#### Ключевые слова, больше всего отрицательно влияющие на дейсвтие пользователя
![image](https://user-images.githubusercontent.com/75137969/187068001-9d937a31-7741-48b7-8f96-9ac749c20e40.png)

#### Ключевые слова, меньше всего влияющие на действие пользователя
![image](https://user-images.githubusercontent.com/75137969/187068062-07a5dfb9-d2c8-4f7a-bc7f-6ddf168f9bae.png)

(hash - переход по какой-либо ссылке)

