# Курсовий Проект: "Дослідження можливості створення паралельного алгоритму розв’язання системи лінійних рівнянь методом Гаусса для випадку квадратної матриці коефіцієнтів"

**Дисципліна**: Технології розподілених систем та паралельні обчислення  
**Університет**: Харківський Національний Університет Імені В. Н. Каразіна

## Вступ

Цей проект досліджує можливість створення паралельного алгоритму для розв’язання систем лінійних рівнянь методом Гаусса, зокрема для квадратних матриць коефіцієнтів. Метою є покращення обчислювальної ефективності методу Гаусса через паралелізацію.

## Мета Проекту

1. **Вивчити метод Гаусса** для розв’язання систем лінійних рівнянь.
2. **Проаналізувати можливість паралелізації** методу Гаусса.
3. **Розробити та реалізувати паралельний алгоритм** для розв’язання систем лінійних рівнянь з квадратною матрицею коефіцієнтів.
4. **Оцінити ефективність** паралельного алгоритму порівняно з посілдовним підходом.

## Огляд Алгоритму

Метод Гаусса є основною технікою для розв’язання систем лінійних рівнянь. Алгоритм складається з трьох основних етапів:

1. **Прямий хід **: Перетворення матриці до верхньої трикутної форми.
2. **Зворотня підстановка або зворотній хід**: Знаходження невідомих.

Паралельна версія алгоритму зосереджується на зменшенні складності обчислень шляхом розподілу обчислювальних етапів серед кількох процесорних одиниць.

## Підхід до Паралелізації

Цей проект досліджує різні стратегії паралелізації:

- **Паралелізм по рядках**: Різні рядки матриці обробляються одночасно.
- **Паралелізм по стовпцях**: Обчислення над стовпцями матриці розподіляються серед кількох процесорів.
- **Розподіл даних**: Розділення матриці на менші блоки для паралельної обробки.

## Технології

- **Мова програмування**: C++/Python
- **Фреймворк для паралельних обчислень**: OpenMPI + CUDA 
- **Представлення матриці**: Власна бібліотека, 2-мірний масив

## Очікувані Результати

1. Детальний аналіз покращення ефективності розв’язання систем лінійних рівнянь за допомогою паралелізованого методу Гаусса.
2. Порівняльні показники ефективності між послідовною та паралельною реалізацією.
3. Робоча реалізація паралельного алгоритму з належною документацією.

## Як запустити

Щоб запустити проект:

1. Клонуйте цей репозиторій.
2. Скомпілюйте код (якщо C++).
3. Запустіть програму з відповідними вхідними файлами для системи лінійних рівнянь.
   mpirun -np <n_proc> ./mpi_program <n_equations> <0 - Gauss / 1 - Jordan-Gauss>
   ./cuda_program <n_equations> <n_thread_per_block>
4. Порівняйте результати між серійною та паралельною реалізацією.

## Ліцензія

Цей проект ліцензовано за ліцензією MIT.
