# monocamera

Программа распознает желтые предметы и находит до них расстояние. Необходимо изначально знать 
ширину предмета. 
При первом запуске, чтобы узнать фокальное расстояние, необходимо изменить строчку в first_run.py


![4-01](https://user-images.githubusercontent.com/101719007/161603735-e453a9d2-e15d-461a-9caf-3900464663a8.jpg)


Расположение файла: 

/home/nvidia/Anna/first_run.py

Открываем файл в текстовом редкторе и изменяем значение на ширину вашего объекта в см, а также расстояние до него:

W = 6.3  #ширина вашего объекта в см

d = 15   #расстояние между объектом и камерой в см

Переменную d мы задаём сами, выставив камеру ровно на это расстояние до предмета.

Сохраняем изменения и запускаем файл.

Запуск: 

Написать в консоли

cd /home/nvidia/Anna/

python monocamera_distance.py

В консоли отобразится значение f - фокусное расстояние. 

![image](https://user-images.githubusercontent.com/101719007/161546059-48828a74-6305-4de5-b14a-a484fe5f7526.png)

Чтобы выйти из программы, нужно нажать 'q'

Теперь имея значение фокального расстояния, мы можем перейти к программе определения расстояния до предмета. 

Расположение файла: 

/home/nvidia/Anna/monocamera_distance.py

Открываем monocamera_distance.py в файловом редакторе и изменяем значение f:

f = 442.8571428571429 #focal length

Запуск: 

Написать в консоли

python monocamera_distance.py

1) Алгоритм работы программы 

![2_Монтажная область 1](https://user-images.githubusercontent.com/101719007/161603133-c97f386e-8372-418b-85cc-818baa67f941.jpg)

(7.04. 2022)
Задача определения местоположения 
![image](https://user-images.githubusercontent.com/101719007/162430042-45f24561-7704-4b85-bc31-42784f5ddc5a.png)

Запуск файдла:

python monocamera_distance3.py

Красными точками отображаюются два знака. 
Зелёной - трек перемещений веб-камеры. 
На данный момент местоположение определяется неточно, так как необходимо расчитать масштаб размера окна в пикселях относительно реального полигона. И соответсвтенно перевести все значения расстояния от объекта до камеры в миллиметры. Также необходимо доработать координатную плоскость (добавить квадраты) и сделать полигон с  соблюдением масштаба.

(11.04.2022)

В комнате поменялось освещение, поэтому пришлось настраивать границы цветов HSV для кругов заново. 

Новые границы для желтого:

![image](https://user-images.githubusercontent.com/101719007/162704746-188f6b4e-9c50-49fb-8c76-ce4a79ae4ed9.png)

Новые границы для зелёного:

![image](https://user-images.githubusercontent.com/101719007/162706023-0160e48c-069e-4c96-9b21-f7a81a589ac8.png)

Пример работы программы monocamera_distance4.py: 

![image](https://user-images.githubusercontent.com/101719007/162742122-99f534b0-e206-45e3-9b93-e16e93b2aa48.png)

Из-за слабой камеры неточно определяется цветовое пятно, что приводит к неточным подсчетам расстояния и координат. 
При наиулучшем положении камеры, когда расстояние определялось верно, погрешность по оси x и y составляла около 2-3 сантиметров. 


Размер полигона: 832 мм на 1150 мм. 

![image](https://user-images.githubusercontent.com/101719007/162813497-38243ea3-5d1c-46d6-9206-c393c6f3028a.png)

![image](https://user-images.githubusercontent.com/101719007/162813468-b350d7aa-9df7-4e82-a4c5-1815bd7d673b.png)

Координата (0,0) находится в верхнем левом углу относительно камеры. 

Было принято решение заменить цветные круги на QR-коды. 

Для распознования QR-кодов необходимо установить библиотеку:

pip install pyzbar

Демонстрация работы программы QR.py

![image](https://user-images.githubusercontent.com/101719007/162758051-93ff5d59-1904-4ee6-b47b-4dad1f5d8971.png)


Демонстрация работы программы monocamera_distance5.py:

![image](https://user-images.githubusercontent.com/101719007/162765328-3f701719-aae9-47cb-ab13-c36fdf50601c.png)

Скачки координат значительно уменьшились, но тест на ноутбуке показал, что коды распознаются намного лучше с веб-камерой с более четкой картинкой.  

19.04.2022

sudo -H pip install -U jetson-stats

sudo jtop

The jetson_stats.service is not active. Please run:

sudo systemctl restart jetson_stats.service

25.04.2022

Был построен полигон.

Размеры полигона: 200см на 360см

![image](https://user-images.githubusercontent.com/101719007/165107704-209fae97-9614-4ad1-b10d-b3ca1b51f872.png)

Длина клетки: 40 см

Добавлены 4 знака. 

Координаты знаков:

ID=1 (280, 160)

ID=2 (280, 40)

ID=3 (360, 140)

ID=4 (360, 80)

Добавлены массивы с координатами знаков, с дистанциями для каждого знака, с координатами для каждой пары знаков. Поиск местоположения будет определяться по среднему арифметическому. Т.е. по итогу работы цикла будут найдены 6 координат и для нахождения финальной координаты разделены на 6. Также возможно использование среднего арифметического взвешенного относительно дистанции для двух знаков.   

11.05.2022

Добавлен фильтр для нейтрализации искажения из-за эффекта рыбьего глаза на веб-камере.

Фильтр работает по следующему принципу:

1) Камера двигалась параллельно знаку и захватывала координаты середины знака на кадре, а также высоту знака в пикселях. Данные, когда знак находится в углу кадра (в этот момент знак искажается больше всего) и когда он посередине, вносятся в таблицу Exel.

2) Всего было получено 1085 значений. Ширина кадра с камеры составляет 1000 пикселей. Будем считать, что идеальная передача знака без искажений находится в середине экрана, 500 по иксу. Следовательно, идеальная высота знака соответсвует координате 500. В данном случае это 69 пикселей. Фрагмент таблицы представлен на следующем фото. 

![image](https://user-images.githubusercontent.com/101719007/167856709-6d6d0015-95b0-4b53-95ab-8536ad43dc54.png)

3) Далее был построен график, где по абциссе расположены координаты середины знака по иксу, а по ординате расположена разница между идеальным значением высоты знака в пикселях и полученным. Получившийся график представлен на следующем фото. 

![image](https://user-images.githubusercontent.com/101719007/167858554-0a46d6b9-44c9-43b7-80c8-ceb2175b1ad6.png)

4) На диаграмме видно, что в полученных данных есть закономерность, но значения разбросаны достаточно хаотично. Была построена полиномиальная линия тренда и получено её уравнение.
5) Программа работает по следующему принципу:
- считается коэффициент из уравнения k=(4*10^(-5))*(cX^2)-0.0287*cX+5.6605, где cX - это координата центра знака по иксу, а k - это отклонение от истинного значения. 
- При подсчете высоты знака из итогового значения дополнительно вычитается полученный коэффициент (k).

Результаты внесённых изменений представлены на следующем фото:

![image](https://user-images.githubusercontent.com/101719007/167860341-d6857454-83c4-4fc4-b96d-8e840e0ebf33.png)

Во-первых, мы видим, что третий и четвёртый знак показывают неверное расстояние до них. 

Во-вторых, значения по "y" для каждого знака усреднились, т.е. уже нет аномальных пар знаков, которые показывают погрешность в 100 см. Однако, как видно на фото, погрешность всё-таки есть. 

В-третьих, Финальная координата дальше реальной по игреку на квадрат (т.е. 40 см). На фото мы видим финальную координату (156, 43), хотя на самом деле реальная координата составляла (160, 80). 

В-четвертых, проблема, когда камеру поворачивали, но она оставалась на той же координате, а программа показывала, что камера движется, частично решена. Заметны улучшения. 

Возможные проблемы:

1) Возможно проблема в измерениях, и необходимо собрать данные для фильтра заново. При измерениях камера перемещалась параллельно знаку, что теоретически может быть ошибкой, так как и расстояние до знака менялось. Следует попробовать зафиксировать камеру на одном месте и, поворачивая, захватывать значения для знака. 
2) Работает ли этот фильтр на всех расстояниях одинаково? 
